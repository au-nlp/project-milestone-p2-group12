import os
from datasets import load_from_disk
from transformers import AutoTokenizer


def load_and_tokenize_data(data_path, model_id, max_source_length=1024, max_target_length=128):
    """
    加载并对数据集进行Tokenize。

    Args:
        data_path (str): 处理好的数据集路径 (例如 'data/tldr_cleaned')
        model_id (str): 模型名称 (例如 'google/flan-t5-base')
        max_source_length (int): 输入文本最大长度
        max_target_length (int): 摘要最大长度

    Returns:
        tokenized_datasets: 处理后的 dataset
        tokenizer: 加载的 tokenizer
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run preprocessing first.")

    print(f"Loading dataset from {data_path}...")
    dataset = load_from_disk(data_path)

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess_function(examples):
        # 这里的键名对应 tldr_cleaned 数据集中的列名
        inputs = examples["post_for_model"]
        targets = examples["gold_tldr"]

        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

        # 为目标（摘要）进行 tokenize
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 批量处理
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,  # 移除原始文本列，只保留 tensor
        desc="Running tokenizer"
    )

    return tokenized_datasets, tokenizer