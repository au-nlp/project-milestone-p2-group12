import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trl import DPOTrainer, DPOConfig

def main():

    #  Path and environment config
    MODEL_PATH = "models/sft_1/checkpoint"  # SFT model path
    DATA_FILE = "data/preferences/dpo_candidates_new.json"  # Preference pair data path
    OUTPUT_DIR = "models/dpo0.01"
    LOG_DIR = f"{OUTPUT_DIR}/logs"

    print("=== Step 5: DPO Training (Strict Config Match) ===")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")



    # Load data and create validation split
    print(f"[Data] Loading dataset from {DATA_FILE}...")
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # Validation split (5% or up to 500 samples)
    val_size = min(500, int(len(full_dataset) * 0.05))
    if val_size < 10: val_size = 10
    
    dataset_split = full_dataset.train_test_split(test_size=val_size, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"[Data] Train Size: {len(train_dataset)} | Eval Size: {len(eval_dataset)}")

    # Load models
    print("[Model] Loading Tokenizer & Policy Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Policy Model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto"
    )
    
    # Reference Model
    print("[Model] Loading Reference Model...")
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        device_map="auto"
    )

    # Training configuration
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="dpo_tldr_final",
        
        # --- Core parameters ---
        beta=0.1,
        loss_type="sigmoid",  
        
        # --- Length limits ---
        max_length=1024,             
        max_prompt_length=1024,      
        max_completion_length=128,   
        
        # --- Optimizer settings ---
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        
        # --- Memory and hardware ---
        gradient_checkpointing=True,
        fp16=True,
        
        # --- Logging and checkpoints ---
        logging_dir=LOG_DIR,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="tensorboard",
        
        # --- Misc ---
        remove_unused_columns=False,
    )

    # Trainer

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Start training
    print("\n>>> Starting DPO Training...")
    trainer.train()

    print(f"\n>>> Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(" Step 5 DPO Training Completed!")

if __name__ == "__main__":
    main()
