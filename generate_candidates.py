import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

# 导入自定义模块
sys.path.append(os.getcwd())
from src.decoding_utils import batch_generate

# 定义要跑的模型列表
MODELS_CONFIG = [
    ("sft", "models/sft_1/checkpoint"), 
    ("bart", "sshleifer/distilbart-cnn-12-6"),
    # ("pegasus", "google/pegasus-xsum")
]

def get_signature(text):
    """生成文本指纹：去除首尾空格，取前100个字符"""
    if not text: return ""
    return text.strip()[:100]

def is_model_processed(responses_dict, model_name):
    """
    【核心修复】
    只要 responses 里包含任何以 "{model_name}_" 开头的 key，
    就认为该模型已经跑过了。
    兼容 'bart_beam5' 和 'bart_beam_search' 等不同命名。
    """
    prefix = f"{model_name}_"
    for k in responses_dict.keys():
        if k.startswith(prefix):
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/tldr_cleaned")
    parser.add_argument("--output_file", type=str, default="data/candidates/train_candidates_hybrid.json")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print("=== Step 2: Generating Candidates (Robust Prefix Match Mode) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 准备数据容器
    final_data_map = {} 
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # --- 断点续传：加载已有数据 ---
    if os.path.exists(args.output_file):
        print(f"Resuming from: {args.output_file}")
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                existing_list = json.load(f)
                for item in existing_list:
                    sig = get_signature(item["prompt"])
                    final_data_map[sig] = item
            print(f"Loaded {len(final_data_map)} samples into memory map.")
        except json.JSONDecodeError:
            print("Warning: JSON corrupted. Starting fresh.")

    # 加载数据集
    dataset = load_from_disk(args.data_path)["train"]
    if args.max_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    # 2. 依次运行模型
    for model_name, model_path in MODELS_CONFIG:
        print(f"\n>>> Processing Model: {model_name}")
        
        # --- 状态预检 (使用前缀匹配) ---
        completed_count = sum(1 for item in final_data_map.values() if is_model_processed(item.get("responses", {}), model_name))
        
        print(f"Status check: {completed_count}/{len(dataset)} items already have {model_name} responses.")
        
        # 如果已经跑够了，跳过
        if completed_count == len(dataset): 
            print(f"Skipping {model_name} (Already complete).")
            continue

        # 加载模型
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            model.eval()
        except Exception as e:
            print(f"Load error: {e}")
            continue

        batch_prompts = []
        batch_sigs = [] 
        batch_meta = [] 
        unsaved_count = 0 
        skipped_count = 0

        pbar = tqdm(total=len(dataset), desc=f"Generating {model_name}")
        
        for i, example in enumerate(dataset):
            prompt = example["post_for_model"]
            sig = get_signature(prompt)
            
            # --- 循环内检查 (使用前缀匹配) ---
            if sig in final_data_map and is_model_processed(final_data_map[sig].get("responses", {}), model_name):
                pbar.update(1)
                skipped_count += 1
                continue
            
            batch_prompts.append(prompt)
            batch_sigs.append(sig)
            batch_meta.append({
                "ref": example["gold_tldr"],
                "plain": example.get("post_plain", "")
            })

            # 执行 Batch 生成
            if len(batch_prompts) == args.batch_size or i == len(dataset) - 1:
                if not batch_prompts: continue
                
                try:
                    results = batch_generate(model, tokenizer, batch_prompts, device)
                    
                    for idx, p_sig in enumerate(batch_sigs):
                        if p_sig not in final_data_map:
                            final_data_map[p_sig] = {
                                "prompt": batch_prompts[idx],
                                "reference": batch_meta[idx]["ref"],
                                "post_plain": batch_meta[idx]["plain"],
                                "responses": {}
                            }
                        
                        for strat, texts in results.items():
                            final_data_map[p_sig]["responses"][f"{model_name}_{strat}"] = texts[idx]
                    
                    unsaved_count += len(batch_prompts)

                except Exception as e:
                    print(f"\n[Error] Batch {i}: {e}")
                
                # 定期保存
                if unsaved_count >= 500 or i == len(dataset) - 1:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(list(final_data_map.values()), f, indent=2)
                    unsaved_count = 0
                
                pbar.update(len(batch_prompts))
                batch_prompts = []
                batch_sigs = []
                batch_meta = []
        
        pbar.close()
        print(f"Skipped {skipped_count} items (Already existed).")
        
        # 最终保存
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(list(final_data_map.values()), f, indent=2)

    print("\nGeneration pipeline completed.")

if __name__ == "__main__":
    main()