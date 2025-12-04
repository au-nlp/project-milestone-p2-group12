import json
import random
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


def get_winner(client, post, sft_pred, dpo_pred, model: str = "gpt-4-turbo") -> str:
    """
    用 GPT-4 对比一对 (SFT, DPO) 摘要，返回 "SFT" / "DPO" / "tie"。

    为避免位置偏差，会随机决定哪一个作为 Summary 1 / Summary 2，
    然后根据返回的 winner 和 is_swapped 还原到 SFT / DPO 的胜负关系。
    """
    # 随机交换顺序：True 表示 DPO 在前，False 表示 SFT 在前
    is_swapped = random.random() > 0.5
    if is_swapped:
        s1, s2 = dpo_pred, sft_pred
    else:
        s1, s2 = sft_pred, dpo_pred

    system_prompt = (
        "You are a strict evaluator of Reddit TL;DR summaries. "
        "Your job is to decide which of two summaries is better for a given post."
    )

    user_prompt = f"""
You will be given a Reddit post and two candidate TL;DR summaries (Summary 1 and Summary 2).

[Original Post]
{post}

[Summary 1]
{s1}

[Summary 2]
{s2}

Evaluate the summaries in the following order of importance:
1. Factuality (CRITICAL): A summary must not hallucinate or contradict the post.
2. Coverage and relevance: Prefer the summary that captures the main point or conflict.
3. Conciseness: If both are factual and relevant, prefer the shorter one.
4. Style: Prefer the one that sounds like a natural Reddit TL;DR (informal but clear).

Constraints:
- Avoid ties. Only output "tie" if the two summaries are essentially identical in content and quality.
- If they are very similar, break the tie in favor of the more concise or more natural summary.

Respond STRICTLY in JSON format:

{{"winner": "1" | "2" | "tie", "reason": "short explanation"}}
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content

        # 处理可能的 ```json 包裹
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
        parsed = json.loads(content.strip())

        w = str(parsed.get("winner", "")).strip().lower()
        if w not in {"1", "2", "tie"}:
            raise ValueError(f"Unexpected winner value: {w}")

        # 先处理平局
        if w == "tie":
            return "tie"

        # 否则根据是否交换还原到 SFT / DPO
        if is_swapped:
            # Summary 1 = DPO, Summary 2 = SFT
            return "DPO" if w == "1" else "SFT"
        else:
            # Summary 1 = SFT, Summary 2 = DPO
            return "SFT" if w == "1" else "DPO"

    except Exception as e:
        print(f"[get_winner] API or parse error: {e}")
        # 出错就当作平局，或者你也可以 return "error" 并在外面跳过
        return "tie"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs_path",
        type=str,
        default="data/metrics/gpt4_eval_pairs.json",
        help="JSON 文件，包含 SFT 和 DPO 的生成结果",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="随机评估多少对样本（上限为文件中实际数量）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo",
        help="用来做评估的 GPT-4 型号，例如 gpt-4-turbo / gpt-4o 等",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key（若不传则从环境变量中读取）",
    )
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    # 读取 SFT / DPO 对比文件
    with open(args.pairs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("pairs_path JSON 必须是 list[dict] 格式。")

    n_available = len(data)
    n_eval = min(args.num_samples, n_available)

    print(f"Running GPT-4 Judge on {n_eval} pairs...")

    # 随机抽样
    samples = random.sample(data, n_eval)

    # 统计结果，统一用小写 'tie'
    results = {"SFT": 0, "DPO": 0, "tie": 0}

    for item in tqdm(samples):
        post = item.get("prompt", "")
        sft_pred = item.get("sft_pred", "")
        dpo_pred = item.get("dpo_pred", "")
        
        winner = get_winner(client, post, sft_pred, dpo_pred, model=args.model)
        if winner in results:
            results[winner] += 1

    total = results["SFT"] + results["DPO"] + results["tie"]
    if total == 0:
        print("No valid results were collected.")
        return

    win_rate = (results["DPO"] + 0.5 * results["tie"]) / total

    print("\n" + "=" * 30)
    print("   HEAD-TO-HEAD WIN RATE   ")
    print("=" * 30)
    print(f"DPO Wins: {results['DPO']}")
    print(f"SFT Wins: {results['SFT']}")
    print(f"Ties:     {results['tie']}")
    print("-" * 30)
    print(f"DPO Win Rate: {win_rate:.2%}")
    print("=" * 30)


if __name__ == "__main__":
    main()
