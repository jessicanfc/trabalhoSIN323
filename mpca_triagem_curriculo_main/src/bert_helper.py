import numpy as np

from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score


def tokenize_resume(examples: LazyBatch, tokenizer: PreTrainedTokenizer, max_len: int) -> BatchEncoding:
    return tokenizer(examples['text'], padding='max_length', max_length=max_len, truncation=True)


def compute_metrics_classification(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    return {
        'f1_macro': f1_macro
    }
