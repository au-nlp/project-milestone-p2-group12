import json
import os
import argparse
from tqdm import tqdm
import sys
import numpy as np

sys.path.append(os.getcwd())
from src.metric_utils import MetricCalculator

# ==============================================================================
# Filtering strategy
# ==============================================================================
WEIGHTS = {"rouge": 0.02, "bert": 0.78, "fact": 0.2}

CONFIG = {
    # Strict factuality threshold: if factuality is too low, directly discard; not eligible to be Chosen
    "factuality_threshold": 0.5,
    
    # Score gap threshold: only when the gap between Chosen and Rejected is large enough, we keep this pair
    "min_score_gap": 0.05, 
    
    # Passing score for human summary
    # If the human-written summary gets a very low score, it indicates the data quality is poor, so we discard the entire item
    "min_human_score": 0.3 
}
# ==============================================================================

def calculate_single_score(metrics_dict, idx, weights):
    """Compute the weighted total score for a single candidate"""
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
        "discarded_human_bad": 0, # Human summary is poor
        "discarded_small_gap": 0  # Scores too close
    }
    
    print(f"Filtering {len(data)} items with strict rules...")
    
    for item in tqdm(data):
        src = item.get("post_plain", "") or item["prompt"]
        human_ref = item["reference"]
        
        # Extract all model-generated candidates
        model_cands = list(item["responses"].values())
        model_strats = list(item["responses"].keys())
        if len(model_cands) < 1: continue
        
        # Include the human summary for scoring as well
        all_texts = model_cands + [human_ref] 
        
        # Batch compute metrics (all model candidates + human)
        # Reference is still the human summary; computing Human vs Human mainly checks Factuality and BERT (self-correlation)
        # This helps us filter out low-quality human summaries that deviate too far from the post (low factuality)
        refs = [human_ref] * len(all_texts)
        srcs = [src] * len(all_texts)
        
        m = calc.compute_batch(all_texts, refs, srcs)
        
        # Collect scores
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
            
        #  Extract roles
        human_item = scored_items[-1]
        model_items = scored_items[:-1]
        
        # Sort model candidates by score
        model_items.sort(key=lambda x: x["score"], reverse=True)
        best_model = model_items[0]
        worst_model = model_items[-1]
        
        stats["processed"] += 1

        # Strategy A: Challenge human (Human vs Model)
        # Only when the human score is passing and clearly higher than the best model
        # do we include this pair (avoid cases where the model is actually better)
        
        # Check human passing line
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
                stats["discarded_small_gap"] += 1
        else:
            stats["discarded_human_bad"] += 1

        # Strategy B: Self-play (Model vs Model)
        gap_b = best_model["score"] - worst_model["score"]
        
        # Only keep if the score gap is large enough and the chosen candidate passes the factuality threshold
        if gap_b >= CONFIG["min_score_gap"]:
            if best_model["fact"] >= CONFIG["factuality_threshold"]:
                # Deduplicate
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
                # Even the best model has too much hallucination; this sample is too noisy, discard it
                pass
        else:
            # Best and worst are too similar, meaning the model's generations for this prompt are very stable
            # (or uniformly bad), so there's little signal to learn from
            pass

    # Save
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
