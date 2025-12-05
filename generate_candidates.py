import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

# Import custom modules
sys.path.append(os.getcwd())
from src.decoding_utils import batch_generate

# ==============================================================================
# Configs
# ==============================================================================
MODELS_CONFIG = [
    # 1. SFT base (style: Reddit informal)
    ("sft", "models/sft"), 
    
    # 2. BART-CNN (style: long, detail-rich summaries)
    ("bart", "sshleifer/distilbart-cnn-12-6"),
    
    # 3. BART-XSum (style: very concise, highly abstract)
    ("bart_xsum", "facebook/bart-large-xsum")
]
# ==============================================================================

def get_signature(text):
    """
    Generate a text fingerprint: strip whitespace and take the first 100 chars.
    Must always generate signatures from the *same raw text* so IDs remain consistent.
    """
    if not text: return ""
    return text.strip()[:100]

def is_model_processed(responses_dict, model_name):
    """
    Check whether this model has already produced outputs.
    """
    prefix = f"{model_name}_"
    for k in responses_dict.keys():
        if k.startswith(prefix):
            return True
    return False

def load_model_safe(model_path, device):
    try:
        # Try standard load
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"  [Warn] Standard load failed, trying with weights_only=False... ({e})")
        # Force loading older weight formats (fixes torch.load errors)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, weights_only=False).to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/tldr_cleaned")
    parser.add_argument("--output_file", type=str, default="data/candidates/train_candidates_new.json")
    parser.add_argument("--max_samples", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=4) 
    args = parser.parse_args()

    print("=== Step 2: Generating Candidates (Fixed Format Mismatch) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Prepare the in-memory data container
    final_data_map = {} 
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # --- Resume from checkpoint if JSON already exists ---
    if os.path.exists(args.output_file):
        print(f"Resuming from: {args.output_file}")
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                existing_list = json.load(f)
                for item in existing_list:
                    # Use the original prompt to reconstruct ID
                    sig = get_signature(item["prompt"])
                    final_data_map[sig] = item
            print(f"Loaded {len(final_data_map)} samples into memory map.")
        except json.JSONDecodeError:
            print("Warning: JSON corrupted. Starting fresh.")

    # Load dataset
    dataset = load_from_disk(args.data_path)["train"]
    if args.max_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    # 2. Run models sequentially
    for model_name, model_path in MODELS_CONFIG:
        print(f"\n>>> Processing Model: {model_name}")
        
        # --- Status check ---
        completed_count = sum(1 for item in final_data_map.values() if is_model_processed(item.get("responses", {}), model_name))
        print(f"Status check: {completed_count}/{len(dataset)} items already have {model_name} responses.")
        
        if completed_count == len(dataset): 
            print(f"Skipping {model_name} (Already complete).")
            continue

        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = load_model_safe(model_path, device)
            model.eval()
        except Exception as e:
            print(f"[Error] Failed to load {model_name}: {e}")
            continue

        batch_prompts = [] # Input fed to the model (may be cleaned)
        batch_sigs = []    # Unique IDs (always based on raw text)
        batch_meta = [] 
        unsaved_count = 0 
        skipped_count = 0

        pbar = tqdm(total=len(dataset), desc=f"Generating {model_name}")
        
        for i, example in enumerate(dataset):
            # Always use raw post_for_model to generate signature.
            # This ensures IDs remain consistent even if model inputs differ.
            raw_prompt_id = example["post_for_model"]
            sig = get_signature(raw_prompt_id)
            
            # Skip if already processed
            if sig in final_data_map and is_model_processed(final_data_map[sig].get("responses", {}), model_name):
                pbar.update(1)
                skipped_count += 1
                continue
            
            # Clean/adjust input (avoid repetition for external models)
            if model_name == "sft":
                # SFT understands the "SUBREDDIT:..." header, so keep as is
                model_input = example["post_for_model"]
            else:
                if "post_body" in example:
                    model_input = example["post_body"]
                else:
                    if "POST:\n" in raw_prompt_id:
                        model_input = raw_prompt_id.split("POST:\n")[-1]
                    else:
                        model_input = raw_prompt_id 

            batch_prompts.append(model_input) 
            batch_sigs.append(sig)
            
            batch_meta.append({
                "raw_prompt": raw_prompt_id,  # Always store the raw formatted prompt
                "ref": example["gold_tldr"],
                "plain": example.get("post_plain", "")
            })

            # Execute batch generation
            if len(batch_prompts) == args.batch_size or i == len(dataset) - 1:
                if not batch_prompts: continue
                
                try:
                    results = batch_generate(model, tokenizer, batch_prompts, device)
                    
                    for idx, p_sig in enumerate(batch_sigs):
                        # Initialize entry
                        if p_sig not in final_data_map:
                            final_data_map[p_sig] = {
                                "prompt": batch_meta[idx]["raw_prompt"],
                                "reference": batch_meta[idx]["ref"],
                                "post_plain": batch_meta[idx]["plain"],
                                "responses": {}
                            }
                        
                        # Save generated results
                        for strat, texts in results.items():
                            final_data_map[p_sig]["responses"][f"{model_name}_{strat}"] = texts[idx]
                    
                    unsaved_count += len(batch_prompts)

                except Exception as e:
                    print(f"\n[Error] Batch {i}: {e}")
                
                # Periodic save
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
        
        # Final save
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(list(final_data_map.values()), f, indent=2)

    print("\nGeneration pipeline completed.")

if __name__ == "__main__":
    main()
