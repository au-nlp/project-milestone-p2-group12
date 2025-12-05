import os
import numpy as np
import torch

from datasets import load_from_disk
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ==========================
# config
# ==========================
MODEL_NAME = "google/flan-t5-base"
DATA_DIR   = "data/tldr_cleaned"      
OUTPUT_DIR = "models/sft"

MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128

LR = 5e-5
EPOCHS = 3
TRAIN_BS = 2
EVAL_BS = 2
GRAD_ACC = 4

rouge = evaluate.load("rouge")


def preprocess_function(examples, tokenizer):
    # source：post_for_model
    model_inputs = tokenizer(
        examples["post_for_model"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
    )
    # target：gold_tldr
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["gold_tldr"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    # decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    result = {k: round(v * 100, 2) for k, v in result.items()}
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 模型 & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 2. 数据集
    dataset = load_from_disk(DATA_DIR)   # 需要包含 train / validation split
    column_names = dataset["train"].column_names

    tokenized = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=column_names,
    )

    # 3. collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    # 4. training args
    bf16_flag = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACC,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        bf16=bf16_flag,
        logging_steps=100,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
