import json
import random
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from openai import OpenAI
import sys

from dotenv import load_dotenv
load_dotenv()
# 确保能导入 src
sys.path.append(os.getcwd())
from src.metric_utils import MetricCalculator

def get_gpt4_judgment(client, post, sum_a, sum_b, model="gpt-4-turbo"):
    # 系统提示词
    system_prompt = """
                    You are an expert evaluator for Reddit TL;DR summarization. 
                    Your goal is to select the summary that best serves as a helpful, accurate, 
                    and concise TL;DR for the original post.
                    """

    # 用户提示词：详细的评估步骤
    user_prompt = f"""
                    Please evaluate two candidate summaries (A and B) for the given Reddit post.

                    [Evaluation Criteria] sorted by importance:
                    1. **Factuality (Critical)**: The summary MUST NOT contain any hallucinations or information contradicting the post. If a summary creates fake details, it MUST lose.
                    2. **Coverage**: The summary should capture the MAIN conflict or question of the post, not just a random detail.
                    3. **Conciseness**: Shorter is better, provided it doesn't lose the core meaning. Reddit TL;DRs should be punchy.
                    4. **Style**: First-person ("I") is preferred if the post is personal.

                    [Reddit Post]:
                    {post}

                    [Summary A]:
                    {sum_a}

                    [Summary B]:
                    {sum_b}

                    [Instructions]:
                    - First, analyze both summaries for factuality errors.
                    - Second, compare which one captures the main point better.
                    - Third, if both are accurate, choose the more concise one.
                    - Finally, output your decision in strictly valid JSON format.

                    Output Format:
                    {{
                        "reason": "Explain step-by-step why one is better than the other...",
                        "winner": "A" (or "B")
                    }}
                """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0 # 保持0温度以获得确定性结果
        )
        
        # 解析返回内容
        content = response.choices[0].message.content
        # 简单的清洗，防止Markdown代码块干扰
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
            
        parsed_json = json.loads(content.strip())
        
        # 打印一下理由（可选，方便你观察模型是怎么想的）
        # print(f"[Reasoning]: {parsed_json.get('reason')...")
        
        return parsed_json.get("winner")
        
    except Exception as e:
        print(f"API Error or JSON Parse Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--input_file", type=str, default="data/candidates/train_candidates_hybrid.json")
    parser.add_argument("--sample_size", type=int, default=50)
    args = parser.parse_args()

    print("=== Step 3: Weight Calibration (LLM Distillation) ===")

    # 1. 采样与 LLM 标注
    # 检查是否有缓存的标注文件，避免重复花钱
    cache_file = "data/calibration/labeled_pairs_cache.json"
    if os.path.exists(cache_file):
        print(f"Loading cached labels from {cache_file}...")
        with open(cache_file, 'r') as f:
            labeled_pairs = json.load(f)
    else:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        valid_data = [d for d in data if len(d.get("responses", {})) >= 2]
        samples = random.sample(valid_data, min(len(valid_data), args.sample_size))
        
        labeled_pairs = []
        client = OpenAI(api_key=args.api_key)
        
        print(">>> Phase 1: Labeling with GPT-4...")
        for i, item in enumerate(samples):
            print(f"Labeling {i+1}/{len(samples)}...", end="\r")
            strats = list(item["responses"].keys())
            k_a, k_b = random.sample(strats, 2)
            winner = get_gpt4_judgment(client, item["prompt"], item["responses"][k_a], item["responses"][k_b])
            
            if winner:
                labeled_pairs.append({
                    "ref": item["reference"],
                    "src": item["post_plain"],
                    "cand_a": item["responses"][k_a],
                    "cand_b": item["responses"][k_b],
                    "label": 1 if winner == "A" else 0
                })
        
        # 保存中间结果！
        os.makedirs("data/calibration", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(labeled_pairs, f, indent=2)
        print(f"\nSaved {len(labeled_pairs)} labeled pairs to {cache_file}")

    # 2. 计算指标矩阵
    print(">>> Phase 2: Computing Metrics...")
    # 显式指定 device，防止和后台运行的生成脚本抢显存
    calc = MetricCalculator() 
    X, y = [], []
    
    for pair in labeled_pairs:
        # A
        m_a = calc.compute_batch([pair["cand_a"]], [pair["ref"]], [pair["src"]])
        # B
        m_b = calc.compute_batch([pair["cand_b"]], [pair["ref"]], [pair["src"]])
        
        diff_r = m_a["rouge"][0] - m_b["rouge"][0]
        diff_b = m_a["bert"][0] - m_b["bert"][0]
        diff_f = m_a["fact"][0] - m_b["fact"][0]
        
        X.append([diff_r, diff_b, diff_f])
        y.append(pair["label"])
        
    X = np.array(X)
    y = np.array(y)

    # 3. 网格搜索最佳权重
    print(">>> Phase 3: Grid Search Optimization...")
    best_acc, best_w = 0, (0,0,0)
    
    # 步长 0.1 遍历
    for r in np.arange(0, 1.1, 0.1):
        for b in np.arange(0, 1.1, 0.1):
            f = 1.0 - r - b
            if f < -0.01: continue
            
            # 预测: 分数 > 0 则 A 赢
            preds = (np.dot(X, [r, b, f]) > 0).astype(int)
            acc = accuracy_score(y, preds)
            
            if acc > best_acc:
                best_acc, best_w = acc, (r, b, f)

    print("\n" + "="*40)
    print(f"Calibration Accuracy: {best_acc:.2%}")
    print(f"Optimal Weights: ROUGE={best_w[0]:.1f}, BERT={best_w[1]:.1f}, FACT={best_w[2]:.1f}")
    print("="*40)
    print("Please update these weights in 4_build_dataset.py config!")

if __name__ == "__main__":
    main()