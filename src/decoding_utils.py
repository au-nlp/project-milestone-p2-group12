# src/decoding_utils.py
import torch

def get_decoding_strategies():
    """
    定义生成策略组合。
    """
    return [
        # 策略1: Beam Search (稳健，质量高，通常作为 Chosen 的强力候选)
        {
            "name": "beam_search",
            "num_beams": 5,
            "do_sample": False,
            "max_new_tokens": 128,
            "min_length": 10,
            "no_repeat_ngram_size": 3,
        },
        # 策略2: Nucleus Sampling (temp=0.7) - 兼顾质量与多样性
        {
            "name": "sampling_mid",
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "max_new_tokens": 128,
        },
        # 策略3: High Temperature (temp=1.0) - 容易出错，作为潜在 Rejected 来源
        {
            "name": "sampling_high",
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 1.0,
            "max_new_tokens": 128,
        }
    ]

def batch_generate(model, tokenizer, batch_prompts, device):
    """
    对一个 Batch 执行所有策略的生成
    """
    strategies = get_decoding_strategies()
    
    # 统一编码
    inputs = tokenizer(
        batch_prompts, 
        padding=True, 
        truncation=True, 
        max_length=1024, # 根据 Reddit 帖子长度调整
        return_tensors="pt"
    ).to(device)

    # 结果字典：{ "beam_search": [text1, text2...], ... }
    batch_results = {conf["name"]: [] for conf in strategies}

    for config in strategies:
        # 过滤掉自定义字段 'name'
        gen_kwargs = {k: v for k, v in config.items() if k != "name"}
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_results[config["name"]] = decoded
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"| WARNING: OOM with strategy {config['name']}. Skipping this batch strategy.")
                torch.cuda.empty_cache()
                # 填充空字符串占位，保持列表长度一致
                batch_results[config["name"]] = [""] * len(batch_prompts)
            else:
                raise e
            
    return batch_results