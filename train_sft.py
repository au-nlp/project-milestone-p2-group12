#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_sft.py

Single-file script for supervised fine-tuning (SFT) a TL;DR summarization model.

Features:
- Load local Hugging Face Dataset from "data/tldr_cleaned"
- Load a Seq2Seq model (default: "google/flan-t5-base")
- Tokenize data and build DataCollator
- Train and evaluate using Seq2SeqTrainer
- Plot loss curves to PNG
- Resume from last checkpoint if available
- Log to TensorBoard under "models/sft/logs"
"""

# ============================================================
# 0. Imports
# ============================================================

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# ============================================================
# 1. Configuration (paths, hyperparameters)
# ============================================================


@dataclass
class TrainConfig:
    """Training configuration and hyperparameters."""
    # Model & data paths (aligned with your train_sft.py)
    model_name: str = "google/flan-t5-base"
    data_dir: str = "data/tldr_cleaned"     # Folder for load_from_disk("data/tldr_cleaned")
    output_dir: str = "models/sft"          # Checkpoints & final model
    log_dir: str = "models/sft/logs"        # TensorBoard logs

    # Input / output lengths
    max_source_len: int = 512
    max_target_len: int = 128

    # Training hyperparameters (roughly aligned with your example)
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01

    # Eval / save strategy (steps-based like your script)
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 100

    # Random seed
    seed: int = 42


# ============================================================
# 2. Model loading utilities
# ============================================================

def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(
    model_name: str,
) -> Tuple[PreTrainedTokenizerBase, AutoModelForSeq2SeqLM, str]:
    """Load tokenizer and Seq2Seq model, move model to the chosen device."""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model, device


# ============================================================
# 3. Dataset loading & preprocessing
# ============================================================

def load_dataset_from_disk(path: str) -> DatasetDict:
    """Load a Hugging Face DatasetDict from disk and perform a simple sanity check."""
    dataset = load_from_disk(path)
    print("Loaded dataset:", dataset)

    if "validation" not in dataset.keys():
        print(
            "Warning: 'validation' split not found in dataset. "
            "Make sure your dataset folder has a 'validation' split."
        )

    return dataset


def preprocess_function(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_source_len: int,
    max_target_len: int,
) -> dict:
    """
    Convert raw fields into model inputs.

    Assumes the dataset contains:
      - 'post_for_model' : the source text
      - 'gold_tldr'      : the target summary
    """
    model_inputs = tokenizer(
        examples["post_for_model"],
        max_length=max_source_len,
        truncation=True,
    )

    # Target text (summary)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["gold_tldr"],
            max_length=max_target_len,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_datasets(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_source_len: int,
    max_target_len: int,
):
    """Apply tokenization to all splits in the DatasetDict."""
    column_names = dataset["train"].column_names

    def _preprocess_batch(batch: dict) -> dict:
        return preprocess_function(
            batch,
            tokenizer=tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
        )

    tokenized = dataset.map(
        _preprocess_batch,
        batched=True,
        remove_columns=column_names,
    )
    return tokenized


# ============================================================
# 4. Evaluation metrics (ROUGE)
# ============================================================

rouge = evaluate.load("rouge")


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    """Return a compute_metrics function for Seq2SeqTrainer."""

    def compute_metrics(eval_pred) -> Dict[str, Any]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 with pad_token_id before decoding labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Compute ROUGE
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: round(v * 100, 2) for k, v in result.items()}

        # Average generated sequence length
        gen_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        result["gen_len"] = float(np.mean(gen_lens))
        return result

    return compute_metrics


# ============================================================
# 5. Helper functions: plotting & inference
# ============================================================

def plot_loss_curve(trainer: Seq2SeqTrainer, output_dir: str) -> None:
    """Plot training and evaluation loss curves from Trainer log_history."""
    history = trainer.state.log_history

    train_points = [
        (h.get("step"), h.get("loss"))
        for h in history
        if "loss" in h and "step" in h
    ]
    eval_points = [
        (h.get("step"), h.get("eval_loss"))
        for h in history
        if "eval_loss" in h and "step" in h
    ]

    # Filter out None values
    train_points = [(s, l) for s, l in train_points if s is not None and l is not None]
    eval_points = [(s, l) for s, l in eval_points if s is not None and l is not None]

    if not train_points and not eval_points:
        print("No loss history found, skipping loss plot.")
        return

    plt.figure()
    if train_points:
        x_tr, y_tr = zip(*train_points)
        plt.plot(x_tr, y_tr, label="train loss")
    if eval_points:
        x_ev, y_ev = zip(*eval_points)
        plt.plot(x_ev, y_ev, label="eval loss")

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved loss curve to {png_path}")


def generate_tldr(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForSeq2SeqLM,
    max_source_len: int,
    max_new_tokens: int = 96,
) -> str:
    """Run inference on a single piece of text using the trained model."""
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# ============================================================
# 6. Main training pipeline
# ============================================================

def main():
    cfg = TrainConfig()

    # Ensure output/log directories exist
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Set random seed for reproducibility
    set_seed(cfg.seed)

    # 1. Load model and tokenizer
    tokenizer, model, device = load_model_and_tokenizer(cfg.model_name)
    print(f"Using device: {device}")
    print(f"Loading dataset from: {cfg.data_dir}")

    # 2. Load dataset and tokenize
    dataset = load_dataset_from_disk(cfg.data_dir)
    tokenized = tokenize_datasets(
        dataset,
        tokenizer=tokenizer,
        max_source_len=cfg.max_source_len,
        max_target_len=cfg.max_target_len,
    )

    # 3. Data collator (with label_pad_token_id=-100 like your script)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    # 4. Metrics
    compute_metrics = build_compute_metrics(tokenizer)

    # 5. Training arguments (aligned with your path & strategy)
    bf16_flag = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,

        # Core hyperparameters
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        weight_decay=cfg.weight_decay,

        # Eval & save strategy (steps-based)
        evaluation_strategy="steps",          # NOTE: correct arg name
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,

        # Misc
        predict_with_generate=True,
        bf16=bf16_flag,
        logging_dir=cfg.log_dir,
        logging_steps=cfg.logging_steps,
        report_to=["tensorboard"],
        generation_max_length=cfg.max_target_len,
        remove_unused_columns=False,
    )

    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Resume-from-checkpoint logic (like your script)
    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        last_checkpoint = get_last_checkpoint(cfg.output_dir)

    print("Starting training...")
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # Plot loss curve
    plot_loss_curve(trainer, cfg.output_dir)

    # Evaluate on test (if present) or validation
    eval_dataset = tokenized.get("test", tokenized["validation"])
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print("Test / validation metrics:", metrics)

    # 8. Save model & tokenizer
    print(f"Saving final model to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # 9. Simple inference example
    if "test" in dataset and len(dataset["test"]) > 0:
        example = dataset["test"][0]["post_for_model"]
    else:
        example = dataset["train"][0]["post_for_model"]

    print(
        "\nSample TL;DR:\n",
        generate_tldr(
            example,
            tokenizer=tokenizer,
            model=model,
            max_source_len=cfg.max_source_len,
        ),
    )

    print(
        "\nTo inspect logs with TensorBoard, run:\n"
        f'  tensorboard --logdir="{cfg.log_dir}"'
    )


# ============================================================
# 7. Entry point
# ============================================================

if __name__ == "__main__":
    main()
