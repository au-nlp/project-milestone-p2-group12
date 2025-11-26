# train_sft.py
import os
import sys
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import get_last_checkpoint

# 导入自定义模块
# 确保 src 目录里有 __init__.py，或者将当前目录加入 path
sys.path.append(os.getcwd())
from src.data_utils import load_and_tokenize_data


def main():
    # --- 配置路径与参数 ---
    MODEL_ID = "google/flan-t5-base"
    DATA_PATH = "data/tldr_cleaned"
    OUTPUT_DIR = "models/sft"
    LOG_DIR = "models/sft/logs"

    # --- 1. 准备数据 ---
    tokenized_datasets, tokenizer = load_and_tokenize_data(
        data_path=DATA_PATH,
        model_id=MODEL_ID
    )

    # --- 2. 加载模型 ---
    print(f"Loading model: {MODEL_ID}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    # --- 3. 训练参数 ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # 核心超参
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # 显存不够可调大此值，减小 batch_size
        weight_decay=0.01,

        # 评估与保存策略 (按 steps 保存，方便断点续训)
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,  # 最多保留3个checkpoint
        load_best_model_at_end=True,

        # 杂项
        predict_with_generate=True,  # 评估时生成摘要计算 ROUGE
        fp16=torch.cuda.is_available(),  # 混合精度
        logging_dir=LOG_DIR,
        logging_steps=100,
        report_to="tensorboard",
    )

    # --- 4. 定义数据整理器 ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100  # 忽略 padding 的 loss
    )

    # --- 5. 初始化 Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 6. 断点续训逻辑 ---
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    print("Starting training...")
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # --- 7. 保存最终模型 ---
    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()