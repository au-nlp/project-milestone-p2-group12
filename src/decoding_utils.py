import torch

def get_decoding_strategies():
    """
    Define generation strategy combinations.
    """
    return [
        # Strategy 1: Beam Search (stable, high quality, usually a strong Chosen candidate)
        {
            "name": "beam_search",
            "num_beams": 5,
            "do_sample": False,
            "max_new_tokens": 128,
            "min_length": 10,
            "no_repeat_ngram_size": 3,
        },
        # Strategy 2: Nucleus Sampling (temp=0.7) - balances quality and diversity
        {
            "name": "sampling_mid",
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "max_new_tokens": 128,
        },
        # Strategy 3: High Temperature (temp=1.0) - more error-prone, used as potential Rejected candidates
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
    Run all strategies for one batch of prompts.
    """
    strategies = get_decoding_strategies()
    
    # Unified tokenization
    inputs = tokenizer(
        batch_prompts, 
        padding=True, 
        truncation=True, 
        max_length=1024, # Adjust according to Reddit post length
        return_tensors="pt"
    ).to(device)

    # Result dict: { "beam_search": [text1, text2...], ... }
    batch_results = {conf["name"]: [] for conf in strategies}

    for config in strategies:
        # Remove custom field 'name'
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
                # Fill empty strings as placeholders to maintain list length
                batch_results[config["name"]] = [""] * len(batch_prompts)
            else:
                raise e
            
    return batch_results
