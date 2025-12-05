import os
from datasets import load_from_disk
from transformers import AutoTokenizer


def load_and_tokenize_data(data_path, model_id, max_source_length=1024, max_target_length=128):
    """
    Load and tokenize the dataset.

    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run preprocessing first.")

    print(f"Loading dataset from {data_path}...")
    dataset = load_from_disk(data_path)

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess_function(examples):
        inputs = examples["post_for_model"]
        targets = examples["gold_tldr"]

        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

        # Tokenize the targets (summaries)
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Batch processing
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,  # Remove original text columns, keep tensors only
        desc="Running tokenizer"
    )

    return tokenized_datasets, tokenizer
