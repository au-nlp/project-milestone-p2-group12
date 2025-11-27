import evaluate
import torch
import numpy as np
from typing import List, Dict

class MetricCalculator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MetricCalculator] Loading metrics on {self.device}...")
        
        # 加载 HF Metrics
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        
        # 加载 SummaC (Factuality)
        # 如果未安装 summac，提供 Mock 降级方案
        self.summac = None
        try:
            from summac.model_summac import SummaCConv
            print("[MetricCalculator] Loading SummaC (VitC)...")
            self.summac = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", device=self.device)
        except ImportError:
            print("[Warning] 'summac' library not found. Using N-gram overlap as mock factuality.")

    def compute_batch(self, candidates: List[str], references: List[str], sources: List[str]) -> Dict[str, List[float]]:
        """
        批量计算 ROUGE, BERTScore, Factuality
        """
        # 1. ROUGE (CPU only, fast)
        # ROUGE-L 是摘要最常用的
        rouge_res = self.rouge.compute(predictions=candidates, references=references, use_aggregator=False)
        r_scores = rouge_res['rougeL']

        # 2. BERTScore (GPU accelerated)
        # 使用 distilbert 提升速度，batch_size 防止显存溢出
        bert_res = self.bertscore.compute(
            predictions=candidates, 
            references=references, 
            lang="en", 
            model_type="distilbert-base-uncased", 
            device=self.device, 
            batch_size=32
        )
        b_scores = bert_res['f1']

        # 3. Factuality
        if self.summac:
            # SummaC 计算 (Source vs Candidate)
            # score 返回的是 logits，通常越高越好
            f_scores = self.summac.score(sources, candidates)['scores']
        else:
            # Mock Implementation: 2-gram Overlap
            f_scores = []
            for src, cand in zip(sources, candidates):
                src_set = set(self._get_ngrams(src, 2))
                cand_set = list(self._get_ngrams(cand, 2))
                if not cand_set:
                    f_scores.append(0.0)
                else:
                    overlap = sum(1 for ng in cand_set if ng in src_set)
                    f_scores.append(overlap / len(cand_set))
        
        return {
            "rouge": r_scores,
            "bert": b_scores,
            "fact": f_scores
        }

    def _get_ngrams(self, text, n):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(n)])