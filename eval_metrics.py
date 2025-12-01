import torch
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

# 复用之前的工具
sys.path.append(os.getcwd())
from src.metric_utils import MetricCalculator

def generate_summaries(model_path, dataset, device, batch_size=8, max_samples=None):
    print(f"Loading model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
        
    predictions = []
    
    # 批量生成
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
        batch = dataset[i : i + batch_size]
        inputs = tokenizer(
            batch["post_for_model"], 
            padding=True, 
            truncation=True, 
            max_length=1024, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # 评估时通常使用 Beam Search 以获得最稳健的结果
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                #开启核采样
                do_sample=True,
                top_p=0.9,
                num_beams=5,
                no_repeat_ngram_size=3
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        
    # 释放显存
    del model
    torch.cuda.empty_cache()
    
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_path", default="models/sft_1/checkpoint")
    parser.add_argument("--dpo_path", default="models/dpo")
    parser.add_argument("--data_path", default="data/tldr_cleaned")
    parser.add_argument("--output_file", default="data/metrics/eval_results.json")
    # 测试集跑500条足够看趋势了，全量跑太慢
    parser.add_argument("--test_samples", type=int, default=500) 
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载测试集
    print("Loading Test Set...")
    dataset = load_from_disk(args.data_path)["test"] # 注意是 test 集！
    
    # 2. 生成摘要
    print("\n>>> Generating SFT summaries...")
    sft_preds = generate_summaries(args.sft_path, dataset, device, max_samples=args.test_samples)
    
    print("\n>>> Generating DPO summaries...")
    dpo_preds = generate_summaries(args.dpo_path, dataset, device, max_samples=args.test_samples)
    
    # 3. 计算指标
    print("\n>>> Computing Metrics...")
    calc = MetricCalculator(device=device)
    
    # 准备 Reference 和 Source
    # 注意：generate_summaries 可能会截断 dataset，所以要重新对齐
    subset = dataset.select(range(len(sft_preds)))
    refs = subset["gold_tldr"]
    srcs = [x.get("post_plain", x["post_for_model"]) for x in subset]
    
    # 计算 SFT 指标
    print("Scoring SFT...")
    m_sft = calc.compute_batch(sft_preds, refs, srcs)
    sft_scores = {
        "rouge": np.mean(m_sft["rouge"]),
        "bert": np.mean(m_sft["bert"]),
        "fact": np.mean(m_sft["fact"])
    }
    
    # 计算 DPO 指标
    print("Scoring DPO...")
    m_dpo = calc.compute_batch(dpo_preds, refs, srcs)
    dpo_scores = {
        "rouge": np.mean(m_dpo["rouge"]),
        "bert": np.mean(m_dpo["bert"]),
        "fact": np.mean(m_dpo["fact"])
    }
    
    # 4. 打印报告
    print("\n" + "="*50)
    print(f"{'Metric':<10} | {'SFT':<10} | {'DPO':<10} | {'Delta':<10}")
    print("-" * 50)
    for k in ["rouge", "bert", "fact"]:
        v1 = sft_scores[k]
        v2 = dpo_scores[k]
        delta = v2 - v1
        print(f"{k.upper():<10} | {v1:.4f}     | {v2:.4f}     | {delta:+.4f}")
    print("="*50)
    
    # 5. 保存结果用于 GPT-4 评测
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 保存生成的对比样本 (JSONL)
    pairs_file = "data/metrics/gpt4_eval_pairs.json"
    pairs_data = []
    for i in range(len(sft_preds)):
        pairs_data.append({
            "prompt": srcs[i],
            "reference": refs[i],
            "sft_pred": sft_preds[i],
            "dpo_pred": dpo_preds[i]
        })
        
    with open(pairs_file, "w") as f:
        json.dump(pairs_data, f, indent=2)
        
    print(f"Saved evaluation pairs to {pairs_file}")

if __name__ == "__main__":
    main()