from typing import Dict, List
from transformers import PreTrainedTokenizerBase
try:
    import evaluate

    rouge_metric = evaluate.load("rouge")
    USE_EVALUATE = True
except Exception:
    from datasets import load_metric

    rouge_metric = load_metric("rouge")
    USE_EVALUATE = False


def build_rouge_metric_fn(tokenizer: PreTrainedTokenizerBase):
    """
    Build a `compute_metrics` function for HuggingFace Trainer.
    It computes ROUGE scores between generated summaries and labels.
    """

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred

        # Some models return a tuple (logits, ...)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Replace -100 in the labels as we can't decode them
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in label_seq]
            for label_seq in labels
        ]
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        # For evaluate.load, result[k] is a Score object; for load_metric it's dict
        cleaned = {}
        for k, v in result.items():
            if hasattr(v, "mid"):
                cleaned[k] = v.mid.fmeasure
            else:
                # datasets.load_metric returns a dict with 'f'
                cleaned[k] = v["f"]

        # Average generated length (in tokens of whitespace-split)
        gen_lens = [len(pred.split()) for pred in decoded_preds]
        cleaned["gen_len"] = sum(gen_lens) / max(len(gen_lens), 1)

        return cleaned

    return compute_metrics


def evaluate_model_on_split(
    model,
    tokenizer: PreTrainedTokenizerBase,
    dataset,
    max_input_length: int = 512,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Standalone evaluation function (not using Trainer).

    - dataset examples must have:
        - "post_for_model" as input
        - "gold_tldr" as reference summary
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def collate_fn(batch) -> (Dict[str, torch.Tensor], List[str]):
        inputs = [ex["post_for_model"] for ex in batch]
        refs = [ex["gold_tldr"] for ex in batch]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return model_inputs, refs

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_preds: List[str] = []
    all_refs: List[str] = []

    with torch.no_grad():
        for model_inputs, refs in dataloader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    result = rouge_metric.compute(
        predictions=all_preds,
        references=all_refs,
        use_stemmer=True,
    )

    cleaned = {}
    for k, v in result.items():
        if hasattr(v, "mid"):
            cleaned[k] = v.mid.fmeasure
        else:
            cleaned[k] = v["f"]

    gen_lens = [len(pred.split()) for pred in all_preds]
    cleaned["gen_len"] = sum(gen_lens) / max(len(gen_lens), 1)

    return cleaned
