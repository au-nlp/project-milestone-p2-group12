import json
import random
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from openai import OpenAI
import sys

sys.path.append(os.getcwd())
from src.metric_utils import MetricCalculator

def get_gpt4_judgment(client, post, sum_a, sum_b, model="gpt-4-turbo"):
    prompt = f"""
    [Task]: Which summary is better for the Reddit post?
    [Criteria]: 1. Factuality (No hallucinations) 2. Conciseness 3. Coherence
    
    [Post]: {post}
    
    [Summary A]: {sum_a}
    
    [Summary B]: {sum_b}
    
    Return strictly JSON: {{"winner": "A"}} or {{"winner": "B"}}
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content).get("winner")
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--input_file", type=str, default="data/candidates/train_candidates_hybrid.json")
    parser.add_argument("--sample_size", type=int, default=50)
    args = parser.parse_args()

    print("=== Step 3: Weight Calibration (LLM Distillation) ===")

    # 1. 采样与 LLM 标注
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # 筛选有效数据
    valid_data = [d for d in data if len(d.get("responses", {})) >= 2]
    samples = random.sample(valid_data, min(len(valid_data), args.sample_size))
    
    labeled_pairs = []
    client = OpenAI(api_key=args.api_key)
    
    print(">>> Phase 1: Labeling with GPT-4...")
    for item in samples:
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

    # 2. 计算指标矩阵
    print(">>> Phase 2: Computing Metrics...")
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