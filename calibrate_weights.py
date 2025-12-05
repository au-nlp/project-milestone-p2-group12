import json
import random
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getcwd())
from src.metric_utils import MetricCalculator


def get_gpt4_judgment(client, post, sum_a, sum_b, model="gpt-4-turbo"):
    """
    用 GPT-4 比较两个摘要（A/B），返回 "A" 或 "B" 作为胜者。
    如果解析失败，返回 None。
    """
    system_prompt = """
You are an expert evaluator for Reddit TL;DR summarization.
Your goal is to select the summary that best serves as a helpful, accurate,
and concise TL;DR for the original post.
"""

    user_prompt = f"""
Please evaluate two candidate summaries (A and B) for the given Reddit post.

[Original Post]
{post}

[Summary A]
{sum_a}

[Summary B]
{sum_b}

[Evaluation Criteria] in order of importance:
1. Factuality (CRITICAL): The summary must not hallucinate or contradict the post.
2. Coverage: It should capture the main point / conflict / question of the post.
3. Clarity: It should be easy to read and understand.
4. Conciseness: Shorter is better if the core meaning is preserved.

[Task]
Choose which summary (A or B) is overall better as a TL;DR according to the criteria.
Respond ONLY with a JSON object of the form:

{{"winner": "A"}}
or
{{"winner": "B"}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        # 有时模型会包一层 ```json ... ```
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
        parsed = json.loads(content.strip())
        winner = parsed.get("winner", "").strip()
        if winner not in ["A", "B"]:
            return None
        return winner
    except Exception as e:
        print(f"API Error or JSON Parse Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API Key",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/candidates/train_candidates_new.json",
        help="Path to candidate file",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Number of examples to sample for calibration",
    )
    parser.add_argument(
        "--force_relabel",
        action="store_true",
        help="Ignore cached labels and call GPT-4 again",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    print("=== Step 3: Weight Calibration (LLM Distillation) ===")

    # ---------- Phase 1: GPT-4 标注 A/B ----------
    cache_file = "data/calibration/labeled_pairs_cache.json"
    use_cache = (os.path.exists(cache_file) and not args.force_relabel)

    if use_cache:
        print(f"Loading cached labels from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            labeled_pairs = json.load(f)
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 只保留至少有两个不同策略响应的样本
        valid_data = [d for d in data if len(d.get("responses", {})) >= 2]
        if len(valid_data) == 0:
            print("No valid samples found in input_file (need >=2 responses each).")
            return

        n_samples = min(len(valid_data), args.sample_size)
        print(f"Sampling {n_samples} examples from {len(valid_data)} valid items...")
        samples = random.sample(valid_data, n_samples)

        labeled_pairs = []
        client = OpenAI(api_key=args.api_key)

        print(">>> Phase 1: Labeling with GPT-4...")
        for i, item in enumerate(samples):
            print(f"Labeling {i+1}/{len(samples)}...", end="\r")
            strats = list(item["responses"].keys())
            # 随机选两个不同策略的候选摘要
            k_a, k_b = random.sample(strats, 2)

            winner = get_gpt4_judgment(
                client,
                item["prompt"],
                item["responses"][k_a],
                item["responses"][k_b],
            )

            if winner:
                labeled_pairs.append(
                    {
                        "ref": item["reference"],
                        "src": item["post_plain"],
                        "cand_a": item["responses"][k_a],
                        "cand_b": item["responses"][k_b],
                        "label": 1 if winner == "A" else 0,
                    }
                )

        print() 
        if len(labeled_pairs) == 0:
            print("GPT-4 labeling failed: no labeled pairs collected. Check API key and network.")
            return

        os.makedirs("data/calibration", exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(labeled_pairs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(labeled_pairs)} labeled pairs to {cache_file}")

    # ---------- Phase 2: 计算三种指标的差值 ----------
    print(">>> Phase 2: Computing Metrics...")
    calc = MetricCalculator()
    X, y = [], []

    for pair in labeled_pairs:
        m_a = calc.compute_batch(
            [pair["cand_a"]], [pair["ref"]], [pair["src"]]
        )
        m_b = calc.compute_batch(
            [pair["cand_b"]], [pair["ref"]], [pair["src"]]
        )

        diff_r = m_a["rouge"][0] - m_b["rouge"][0]
        diff_b = m_a["bert"][0] - m_b["bert"][0]
        diff_f = m_a["fact"][0] - m_b["fact"][0]

        X.append([diff_r, diff_b, diff_f])
        y.append(pair["label"])

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] == 0:
        print("No metric data available for calibration.")
        return

    # 单指标 accuracy 看一下大致上限
    def metric_acc(diff, y, name):
        preds = (diff > 0).astype(int)  # Δmetric > 0 → 预测 A
        acc = accuracy_score(y, preds)
        print(f"{name} only accuracy: {acc:.2%}")

    print(">>> Single-metric accuracies (A wins if Δmetric > 0):")
    metric_acc(X[:, 0], y, "ROUGE")
    metric_acc(X[:, 1], y, "BERT")
    metric_acc(X[:, 2], y, "FACT")

    # ---------- Phase 3A: Logistic Regression 学线性权重（分析用） ----------
    print(">>> Phase 3A: Logistic Regression for linear weights...")

    if len(np.unique(y)) < 2:
        print("Only one class in labels; Logistic Regression cannot be fit. Skipping LR.")
        lr_info = None
    else:
        clf = LogisticRegression(
            penalty="l2",
            C=1e6,             # 几乎不做正则
            fit_intercept=False,
            solver="lbfgs",
            max_iter=1000,
        )
        clf.fit(X, y)

        raw_w = clf.coef_[0]  # 可能有正有负
        # 用 decision_function 看原始线性分数
        raw_preds = (clf.decision_function(X) > 0).astype(int)
        raw_acc = accuracy_score(y, raw_preds)

        # 投影到 simplex：非负 + 和为 1
        w_proj = np.maximum(raw_w, 0.0)
        if w_proj.sum() <= 0:
            w_proj = np.ones_like(w_proj) / len(w_proj)
        else:
            w_proj = w_proj / w_proj.sum()

        proj_preds = (np.dot(X, w_proj) > 0).astype(int)
        proj_acc = accuracy_score(y, proj_preds)

        print(f"LR raw weights (ROUGE, BERT, FACT): {raw_w}")
        print(f"LR raw decision accuracy: {raw_acc:.2%}")
        print(
            "LR projected weights (ROUGE, BERT, FACT): "
            f"({w_proj[0]:.3f}, {w_proj[1]:.3f}, {w_proj[2]:.3f})"
        )
        print(f"LR projected decision accuracy: {proj_acc:.2%}")

        lr_info = (w_proj, proj_acc)

    # ---------- Phase 3B: 在 simplex 上 Grid Search（最终权重推荐） ----------
    print(">>> Phase 3B: Grid Search on simplex for metric weights...")

    def eval_acc(weights, X, y):
        preds = (np.dot(X, weights) > 0).astype(int)
        return accuracy_score(y, preds)

    best_acc = 0.0
    best_w = None

    step = 0.01  # 网格步长
    r_values = np.arange(0.0, 1.0 + 1e-9, step)

    for r in r_values:
        max_b = 1.0 - r
        b_values = np.arange(0.0, max_b + 1e-9, step)
        for b in b_values:
            f = 1.0 - r - b
            if f < 0:
                continue

            w = np.array([r, b, f])
            acc = eval_acc(w, X, y)
            if acc > best_acc:
                best_acc = acc
                best_w = w

    print("\n" + "=" * 40)
    print(f"Grid Search Accuracy : {best_acc:.2%}")
    print(
        "Best Weights from Grid (ROUGE, BERT, FACT) = "
        f"({best_w[0]:.3f}, {best_w[1]:.3f}, {best_w[2]:.3f})"
    )
    if lr_info is not None:
        print("-" * 40)
        print(
            "LR projected weights (for comparison) = "
            f"({lr_info[0][0]:.3f}, {lr_info[0][1]:.3f}, {lr_info[0][2]:.3f}), "
            f"acc = {lr_info[1]:.2%}"
        )
    print("=" * 40)
    print("Please update these weights in 4_build_dataset.py config!")


if __name__ == "__main__":
    main()
