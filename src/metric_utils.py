import evaluate
import torch
import numpy as np
from sentence_transformers import CrossEncoder

class MetricCalculator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Metrics] Loading metrics on {self.device}...")
        
        # Load ROUGE and BERTScore
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        
        # cross-encoder/nli-distilroberta-base
        print("[Metrics] Loading Cross-Encoder for Factuality...")
        self.fact_model = CrossEncoder('cross-encoder/nli-deberta-v3-base', device=self.device)

    def _score_factuality(self, sources, candidates):
        """
        Compute factual consistency score (Entailment Probability)
        """
        # Construct input pairs: [(source, candidate1), (source, candidate2)...]
        # Note: CrossEncoder will automatically handle truncation (max_length=512)
        pairs = list(zip(sources, candidates))
        
        # Predict logits -> softmax -> probabilities
        # model outputs 3 classes: [contradiction, entailment, neutral] (typical order)
        # Or sometimes [contradiction, entailment, neutral]
        # Check model card: label 0=contradiction, 1=entailment, 2=neutral
        scores = self.fact_model.predict(pairs, apply_softmax=True)
        
        # Take the probability of Entailment as the factuality score
        # Index 1 corresponds to Entailment
        entailment_scores = scores[:, 1]
        
        return entailment_scores.tolist()

    def compute_batch(self, candidates, references, sources):
        """
        Compute all metrics in batch
        """
        # 1. ROUGE (CPU)
        r_res = self.rouge.compute(predictions=candidates, references=references, use_aggregator=False)
        r_scores = r_res['rougeL']
        
        # 2. BERTScore (GPU)
        b_res = self.bertscore.compute(
            predictions=candidates, references=references, lang="en", 
            model_type="distilbert-base-uncased", device=self.device, batch_size=32
        )
        b_scores = b_res['f1']
        
        # 3. Factuality (Cross-Encoder)
        f_scores = self._score_factuality(sources, candidates)
            
        return {"rouge": r_scores, "bert": b_scores, "fact": f_scores}
