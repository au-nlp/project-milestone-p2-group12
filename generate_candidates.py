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

# ==============================================================================
# [配置区] 定义模型列表
# ==============================================================================
# 建议使用 bart-large-xsum 替代 pegasus，速度快且效果好
MODELS_CONFIG = [
    # 1. SFT 基座 (风格：Reddit 口语)'
    ("sft", "models/sft_1/checkpoint"), 
    
    # 2. BART-CNN (风格：长文本，细节丰富)
    ("bart", "sshleifer/distilbart-cnn-12-6"),
    
    # 3. BART-XSum (风格：极简，高度概括) 
    ("bart_xsum", "facebook/bart-large-xsum")
]
# ==============================================================================

def get_signature(text):
    """
    生成文本指纹：去除首尾空格，取前100个字符。
    【关键】必须始终对‘同一段文本’生成签名，保证ID一致。
    """
    if not text: return ""
    return text.strip()[:100]

def is_model_processed(responses_dict, model_name):
    """
    检查该模型是否已经生成过
    """
    prefix = f"{model_name}_"
    for k in responses_dict.keys():
        if k.startswith(prefix):
            return True
    return False

def load_model_safe(model_path, device):
    """
    【修复】安全加载模型，兼容旧版权重格式 (.bin)
    """
    try:
        # 尝试默认加载
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"  [Warn] Standard load failed, trying with weights_only=False... ({e})")
        # 强制允许加载旧格式权重 (解决 torch.load 报错)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, weights_only=False).to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/tldr_cleaned")
    parser.add_argument("--output_file", type=str, default="data/candidates/train_candidates_new.json")
    parser.add_argument("--max_samples", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=4) # 显存不够改成 1 或 2
    args = parser.parse_args()

    print("=== Step 2: Generating Candidates (Fixed Format Mismatch) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
                    # 使用原始 prompt 恢复 ID
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
        
        # --- 状态预检 ---
        completed_count = sum(1 for item in final_data_map.values() if is_model_processed(item.get("responses", {}), model_name))
        print(f"Status check: {completed_count}/{len(dataset)} items already have {model_name} responses.")
        
        if completed_count == len(dataset): 
            print(f"Skipping {model_name} (Already complete).")
            continue

        # 加载模型与分词器
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = load_model_safe(model_path, device)
            model.eval()
        except Exception as e:
            print(f"[Error] Failed to load {model_name}: {e}")
            continue

        batch_prompts = [] # 实际喂给模型的输入 (可能清洗过)
        batch_sigs = []    # 数据的唯一 ID (始终基于原始文本)
        batch_meta = [] 
        unsaved_count = 0 
        skipped_count = 0

        pbar = tqdm(total=len(dataset), desc=f"Generating {model_name}")
        
        for i, example in enumerate(dataset):
            # -------------------------------------------------------
            # [核心修复 1] 始终使用原始 post_for_model 生成 ID (Signature)
            # -------------------------------------------------------
            # 这样无论喂给模型什么内容，ID 永远指向同一个条目，保证数据能合并
            raw_prompt_id = example["post_for_model"]
            sig = get_signature(raw_prompt_id)
            
            # 检查是否已处理
            if sig in final_data_map and is_model_processed(final_data_map[sig].get("responses", {}), model_name):
                pbar.update(1)
                skipped_count += 1
                continue
            
            # -------------------------------------------------------
            # [核心修复 2] 针对外部模型清洗输入，防止复读
            # -------------------------------------------------------
            if model_name == "sft":
                # SFT 模型微调过，认识 "SUBREDDIT:..." 头，所以保留
                model_input = example["post_for_model"]
            else:
                # BART/Pegasus 没见过 header，直接喂正文 (post_body)
                # 如果 dataset 里没有 post_body，就手动切割
                if "post_body" in example:
                    model_input = example["post_body"]
                else:
                    # 兜底：去掉 SUBREDDIT/TITLE 头
                    if "POST:\n" in raw_prompt_id:
                        model_input = raw_prompt_id.split("POST:\n")[-1]
                    else:
                        model_input = raw_prompt_id # 实在没办法才用原始的

            batch_prompts.append(model_input) # 喂给模型的
            batch_sigs.append(sig)            # 用来存贮的 ID
            
            batch_meta.append({
                "raw_prompt": raw_prompt_id,  # 始终保存原始 Prompt 到 JSON，方便 DPO 训练
                "ref": example["gold_tldr"],
                "plain": example.get("post_plain", "")
            })

            # 执行 Batch 生成
            if len(batch_prompts) == args.batch_size or i == len(dataset) - 1:
                if not batch_prompts: continue
                
                try:
                    # 调用生成 (传入的是清洗过的 prompts)
                    results = batch_generate(model, tokenizer, batch_prompts, device)
                    
                    for idx, p_sig in enumerate(batch_sigs):
                        # 初始化条目
                        if p_sig not in final_data_map:
                            final_data_map[p_sig] = {
                                "prompt": batch_meta[idx]["raw_prompt"], # 保存原始带格式的 Prompt
                                "reference": batch_meta[idx]["ref"],
                                "post_plain": batch_meta[idx]["plain"],
                                "responses": {}
                            }
                        
                        # 写入结果
                        for strat, texts in results.items():
                            final_data_map[p_sig]["responses"][f"{model_name}_{strat}"] = texts[idx]
                    
                    unsaved_count += len(batch_prompts)

                except Exception as e:
                    print(f"\n[Error] Batch {i}: {e}")
                
                # 定期保存
                if unsaved_count >= 200 or i == len(dataset) - 1:
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