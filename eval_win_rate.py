import json
import random
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def get_winner(client, post, pred_a, pred_b):
    # 随机交换顺序
    is_swapped = random.random() > 0.5
    s1, s2 = (pred_b, pred_a) if is_swapped else (pred_a, pred_b)
    
    # === 严厉版 Prompt ===
    prompt = f"""
    You are a critical editor evaluating Reddit TL;DR summaries. 
    Your goal is to distinguish the better summary, even if the difference is subtle.
    
    [Input Post]:
    {post}
    
    [Summary 1]:
    {s1}
    
    [Summary 2]:
    {s2}
    
    [Evaluation Steps]:
    1. **Factuality Check**: Does either summary contain a hallucination? If yes, it loses immediately.
    2. **Information Density**: If both are factual, which one conveys the SAME meaning with FEWER words? The more concise one wins.
    3. **Tone Check**: Which one sounds more like a human Redditor (casual, direct) and less like a robot?
    4. **Redundancy**: Does one summary repeat information unnecessarily? If so, it loses.

    [Constraints]:
    - **AVOID TIES**: You must try your best to pick a winner. 
    - Only output "tie" if the two summaries are semantically IDENTICAL.
    - If Summary 1 and 2 are very similar, pick the one that is slightly more concise or natural.
    - But if they are truly equal in quality, output "tie".
    Return strictly JSON: {{"winner": "1", "reason": "short explanation"}} OR {{"winner": "2", ...}} OR {{"winner": "tie"}}
    """
    
    try:
        res = client.chat.completions.create(
            model="gpt-4-turbo", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0 # 保持0，减少随机性
        )
        content = res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        w = str(parsed.get("winner", "")).lower()
        
        # 打印理由，方便你分析（可选）
        # print(f"Reason: {parsed.get('reason')}")

        if "tie" in w: return "tie"
        
        # 还原位置
        if is_swapped:
            return "DPO" if "1" in w else "SFT"
        else:
            return "SFT" if "1" in w else "DPO"
            
    except Exception as e:
        print(f"Error: {e}")
        return "error"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/metrics/gpt4_eval_pairs.json")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--sample_size", type=int, default=10) # 测50-100条即可
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print("Please run 6_eval_metrics.py first!")
        return

    with open(args.input_file) as f: data = json.load(f)
    samples = random.sample(data, min(len(data), args.sample_size))
    
    client = OpenAI(api_key=args.api_key)
    results = {"SFT": 0, "DPO": 0, "Tie": 0}
    
    print(f"Running GPT-4 Judge on {len(samples)} pairs...")
    for item in tqdm(samples):
        winner = get_winner(client, item["prompt"], item["sft_pred"], item["dpo_pred"])
        if winner in results:
            results[winner] += 1
            
    total = results["SFT"] + results["DPO"] + results["Tie"]
    win_rate = (results["DPO"] + 0.5 * results["Tie"]) / total
    
    print("\n" + "="*30)
    print("   HEAD-TO-HEAD WIN RATE   ")
    print("="*30)
    print(f"DPO Wins: {results['DPO']}")
    print(f"SFT Wins: {results['SFT']}")
    print(f"Ties:     {results['Tie']}")
    print("-" * 30)
    print(f"DPO Win Rate: {win_rate:.2%}")
    print("="*30)

if __name__ == "__main__":
    main()