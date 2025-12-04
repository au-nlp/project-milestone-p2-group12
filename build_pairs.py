# Results from calibrate_weights.py : 
# ==================================================================
# Computing metrics for pair 400/400...
# >>> Single-metric accuracies (A wins if Δmetric > 0):
# ROUGE only accuracy: 58.50%
# BERT only accuracy: 58.75%
# FACT only accuracy: 63.50%
# >>> Phase 3A: Logistic Regression for linear weights...
# LR raw weights (ROUGE, BERT, FACT): [2.81564485 7.07069005 1.2073623 ]
# LR raw decision accuracy: 65.25%
# LR projected weights (ROUGE, BERT, FACT): (0.254, 0.637, 0.109)
# LR projected decision accuracy: 65.25%
# >>> Phase 3B: Grid Search on simplex for metric weights...
# ==================================================================
# Grid Search Accuracy : 67.50%
# Best Weights from Grid (ROUGE, BERT, FACT) = (0.020, 0.780, 0.200)
# ==================================================================
import json
import os
import argparse
from tqdm import tqdm
import sys
import numpy as np

sys.path.append(os.getcwd())
from src.metric_utils import MetricCalculator

# ==============================================================================
# 筛选策略
# ==============================================================================
WEIGHTS = {"rouge": 0.02, "bert": 0.78, "fact": 0.2}

CONFIG = {
    # 严格的事实性门槛：如果事实性太差，直接踢出，没资格做 Chosen
    "factuality_threshold": 0.5,
    
    # 分差阈值：只有当 Chosen 和 Rejected 差距足够大，才收录这对
    "min_score_gap": 0.05, 
    
    # 人类摘要的及格线
    # 如果人类写的摘要算出来分数太低，说明这个数据本身质量差，整条丢弃
    "min_human_score": 0.3 
}
# ==============================================================================

def calculate_single_score(metrics_dict, idx, weights):
    """计算单个候选的加权总分"""
    return (
        metrics_dict["rouge"][idx] * weights["rouge"] +
        metrics_dict["bert"][idx]  * weights["bert"] +
        metrics_dict["fact"][idx]  * weights["fact"]
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data_backup/candidates/train_candidates_new.json")
    parser.add_argument("--output_file", default="data/preferences/dpo_candidates_new.json") 
    args = parser.parse_args()

    print("=== Step 4: Building High-Quality Preference Dataset ===")
    
    calc = MetricCalculator()
    with open(args.input_file, 'r') as f: data = json.load(f)
    
    final_pairs = []
    stats = {
        "processed": 0,
        "kept_type_a": 0, # Human vs Model
        "kept_type_b": 0, # Model vs Model
        "discarded_human_bad": 0, # 人类写得烂
        "discarded_small_gap": 0  # 分不开
    }
    
    print(f"Filtering {len(data)} items with strict rules...")
    
    for item in tqdm(data):
        src = item.get("post_plain", "") or item["prompt"]
        human_ref = item["reference"]
        
        # 提取所有模型生成的候选
        model_cands = list(item["responses"].values())
        model_strats = list(item["responses"].keys())
        if len(model_cands) < 1: continue
        
        # -------------------------------------------------------
        # 1. 统一大乱斗：把 Human 也放进去一起打分
        # -------------------------------------------------------
        all_texts = model_cands + [human_ref] # 最后一个是 Human
        
        # 批量计算 (所有模型候选 + Human)
        # 注意：这里 Reference 依然用 Human，计算 Human vs Human 的分数主要看 Factuality 和 BERT(自相关)
        # 这能帮我们筛出那些人类写得离题太远（低 Factuality）的脏数据
        refs = [human_ref] * len(all_texts)
        srcs = [src] * len(all_texts)
        
        m = calc.compute_batch(all_texts, refs, srcs)
        
        # 整理分数
        scored_items = []
        for i in range(len(all_texts)):
            score = calculate_single_score(m, i, WEIGHTS)
            fact = m["fact"][i]
            
            is_human = (i == len(all_texts) - 1)
            strategy = "human_gold" if is_human else model_strats[i]
            
            scored_items.append({
                "text": all_texts[i],
                "score": score,
                "fact": fact,
                "type": "human" if is_human else "model",
                "strategy": strategy
            })
            
        # -------------------------------------------------------
        # 2. 提取角色
        # -------------------------------------------------------
        human_item = scored_items[-1]
        model_items = scored_items[:-1]
        
        # 按分数对模型候选排序
        model_items.sort(key=lambda x: x["score"], reverse=True)
        best_model = model_items[0]
        worst_model = model_items[-1]
        
        stats["processed"] += 1

        # -------------------------------------------------------
        # 3. 构建策略 A: 挑战人类 (Human vs Model)
        # -------------------------------------------------------
        # 只有当人类的分数 合格，且 显著高于 最好模型时，才收录
        # (避免模型明明写得比人好，却被强行判负)
        
        # 3.1 检查人类及格线
        if human_item["score"] >= CONFIG["min_human_score"]:
            gap_a = human_item["score"] - best_model["score"]
            
            if gap_a >= CONFIG["min_score_gap"]:
                final_pairs.append({
                    "prompt": item["prompt"],
                    "chosen": human_item["text"],
                    "rejected": best_model["text"],
                    "type": "human_vs_model",
                    "score_gap": gap_a
                })
                stats["kept_type_a"] += 1
            else:
                # 这种情况下，模型已经逼近人类，甚至反超。
                # 激进策略：如果模型分比人高，反向构建 (Model > Human)？
                # 保守策略：丢弃 (神仙打架，不学了) -> 这里选丢弃，求稳
                stats["discarded_small_gap"] += 1
        else:
            stats["discarded_human_bad"] += 1

        # -------------------------------------------------------
        # 4. 构建策略 B: 自我博弈 (Model vs Model)
        # -------------------------------------------------------
        gap_b = best_model["score"] - worst_model["score"]
        
        # 只有分差够大，且 Chosen 的事实性没有硬伤
        if gap_b >= CONFIG["min_score_gap"]:
            if best_model["fact"] >= CONFIG["factuality_threshold"]:
                # 去重
                if best_model["text"] != worst_model["text"]:
                    final_pairs.append({
                        "prompt": item["prompt"],
                        "chosen": best_model["text"],
                        "rejected": worst_model["text"],
                        "type": "model_vs_model",
                        "score_gap": gap_b
                    })
                    stats["kept_type_b"] += 1
            else:
                # 最好模型的幻觉都太多，这组数据太烂，不要了
                pass
        else:
            # 最好和最差差不多，说明模型对这个prompt生成的都很稳（或者都很烂），学不到东西
            pass

    # 保存
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(final_pairs, f, indent=2)
        
    print("\n" + "="*40)
    print("       DATASET CLEANING REPORT       ")
    print("="*40)
    print(f"Original Prompts:      {len(data)}")
    print(f"Human Refs Discarded:  {stats['discarded_human_bad']} (Low Quality)")
    print(f"Pairs Discarded:       {stats['discarded_small_gap']} (Too Similar)")
    print("-" * 40)
    print(f"Final Dataset Size:    {len(final_pairs)}")
    print(f"  - Type A (Human>AI): {stats['kept_type_a']}")
    print(f"  - Type B (AI > AI):  {stats['kept_type_b']}")
    print("="*40)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()