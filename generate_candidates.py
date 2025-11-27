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
    # 请确保这个路径下包含 config.json 和 model.safetensors/bin
    ("sft", "models/sft_1/checkpoint"), 
    ("bart", "facebook/bart-large-cnn"),
    ("pegasus", "google/pegasus-xsum")
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/tldr_cleaned")
    parser.add_argument("--output_file", type=str, default="data/candidates/train_candidates_hybrid.json")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print("=== Step 2: Generating Candidates (Hybrid Strategy) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 准备数据容器
    final_data = {}
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 断点续传：加载已有数据
    if os.path.exists(args.output_file):
        print(f"Resuming from existing file: {args.output_file}")
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                existing_list = json.load(f)
                for item in existing_list:
                    final_data[item["prompt"]] = item
            print(f"Loaded {len(final_data)} existing samples.")
        except json.JSONDecodeError:
            print("Warning: Existing file is corrupted or empty. Starting fresh.")

    # 加载数据集
    dataset = load_from_disk(args.data_path)["train"]
    if args.max_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    # 2. 依次运行每个模型
    for model_name, model_path in MODELS_CONFIG:
        print(f"\n>>> Processing Model: {model_name} from {model_path}")
        
        # 路径检查
        if not os.path.exists(model_path) and not model_path.count("/") <= 1: 
            # 简单的检查：如果是本地路径且不存在，报错；如果是 HF ID (如 facebook/bart...) 则跳过检查
            print(f"Error: Model path {model_path} not found!")
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            model.eval()
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}. Skipping.")
            continue

        batch_prompts = []
        batch_meta = [] 
        
        # 用于保存的计数器
        unsaved_count = 0 

        # 进度条
        pbar = tqdm(total=len(dataset), desc=f"Generating {model_name}")
        
        for i, example in enumerate(dataset):
            prompt = example["post_for_model"]
            
            # --- [修复1] 细粒度的跳过逻辑 ---
            # 检查当前 prompt 是否已经有该模型生成的 key (例如 sft_beam5)
            # 如果 final_data 里有这个 prompt，且 responses 里已经有 sft_beam5，说明跑过了
            test_key = f"{model_name}_beam5" 
            if prompt in final_data and test_key in final_data[prompt].get("responses", {}):
                pbar.update(1)
                continue
            
            batch_prompts.append(prompt)
            batch_meta.append({
                "ref": example["gold_tldr"],
                "plain": example.get("post_plain", "")
            })

            # 执行 Batch 生成
            if len(batch_prompts) == args.batch_size or i == len(dataset) - 1:
                if not batch_prompts: 
                    continue
                
                try:
                    # 生成
                    results = batch_generate(model, tokenizer, batch_prompts, device)
                    
                    # 写入内存字典
                    for idx, p in enumerate(batch_prompts):
                        if p not in final_data:
                            final_data[p] = {
                                "prompt": p,
                                "reference": batch_meta[idx]["ref"],
                                "post_plain": batch_meta[idx]["plain"],
                                "responses": {}
                            }
                        
                        # 存入结果
                        for strat, texts in results.items():
                            final_data[p]["responses"][f"{model_name}_{strat}"] = texts[idx]
                    
                    unsaved_count += len(batch_prompts)

                except Exception as e:
                    print(f"\n[Error] Generation failed at batch {i}: {e}")
                    # 不中断，继续下一个 batch
                
                # --- [修复2] 定期保存 (每500条或最后) ---
                if unsaved_count >= 500 or i == len(dataset) - 1:
                    print(f"\nSaving progress... ({len(final_data)} items)")
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(list(final_data.values()), f, indent=2)
                    unsaved_count = 0
                
                pbar.update(len(batch_prompts))
                
                # 清空 batch
                batch_prompts = []
                batch_meta = []
        
        pbar.close()
        
        # 模型跑完后的最终保存
        print(f"Finished {model_name}. Final save...")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(list(final_data.values()), f, indent=2)

    print("\nGeneration pipeline completed.")

if __name__ == "__main__":
    main()