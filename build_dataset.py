import json
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.getcwd())
from src.metric_utils import MetricCalculator

# ==========================================
# 配置: 请填入 Step 3 跑出来的最佳权重
# ==========================================
CONFIG = {
    "weights": {"rouge": 0.1, "bert": 0.6, "fact": 0.3},
    "factuality_threshold": 0.5,  # 事实性低于此值的摘要直接丢弃
    "min_score_gap": 0.05         # 机器vs机器 的最小分差
}

def calculate_score(metrics_dict, i, weights):
    return (
        metrics_dict["rouge"][i] * weights["rouge"] +
        metrics_dict["bert"][i]  * weights["bert"] +
        metrics_dict["fact"][i]  * weights["fact"]
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/candidates/train_candidates_hybrid.json")
    parser.add_argument("--output_file", default="data/preferences/dpo_mixed_data.json")
    args = parser.parse_args()

    print("=== Step 4: Building Preference Dataset ===")
    
    if not os.path.exists(args.input_file):
        print("Input file not found.")
        return

    calc = MetricCalculator()
    with open(args.input_file, 'r') as f: data = json.load(f)
    
    final_pairs = []
    
    # 遍历每个 Prompt
    for item in tqdm(data):
        ref = item["reference"]
        src = item.get("post_plain", "") or item["prompt"]
        
        # 1. 准备数据进行批量打分
        candidates_list = []
        strategy_list = []
        
        for strat, text in item["responses"].items():
            if not text.strip(): continue
            candidates_list.append(text)
            strategy_list.append(strat)
            
        if len(candidates_list) < 2: continue
        
        # 2. 批量计算指标
        metrics = calc.compute_batch(candidates_list, [ref]*len(candidates_list), [src]*len(candidates_list))
        
        # 3. 综合打分
        scored_cands = []
        for i in range(len(candidates_list)):
            # 事实性过滤
            if metrics["fact"][i] < CONFIG["factuality_threshold"]:
                continue
            
            score = calculate_score(metrics, i, CONFIG["weights"])
            scored_cands.append({
                "text": candidates_list[i],
                "score": score,
                "strategy": strategy_list[i]
            })
            
        if not scored_cands: continue
        
        # 排序
        scored_cands.sort(key=lambda x: x["score"], reverse=True)
        best = scored_cands[0]
        worst = scored_cands[-1]
        
        # --- 类型 A: Human vs Model ---
        # 认为 Human (Ref) 总是最好的
        final_pairs.append({
            "prompt": item["prompt"],
            "chosen": ref,
            "rejected": best["text"],
            "type": "human_vs_model"
        })
        
        # --- 类型 B: Model vs Model ---
        if best["score"] - worst["score"] >= CONFIG["min_score_gap"]:
            # 去重
            if best["text"] != worst["text"]:
                final_pairs.append({
                    "prompt": item["prompt"],
                    "chosen": best["text"],
                    "rejected": worst["text"],
                    "type": "model_vs_model"
                })

    # 保存
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(final_pairs, f, indent=2)
        
    print(f"Done! Generated {len(final_pairs)} pairs. Saved to {args.output_file}")

if __name__ == "__main__":
    main()