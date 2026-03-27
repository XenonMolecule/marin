"""Lightweight mediqa_qa2019 utils — ROUGE-only, no bleurt/bert-score dependency."""

import numpy as np

try:
    import evaluate

    rouge = evaluate.load("rouge")
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("Please install: pip install evaluate rouge_score>=0.1.2")


def doc_to_text(doc) -> str:
    return doc["QUESTION"]["QuestionText"]


def doc_to_target(doc) -> str:
    return doc["QUESTION"]["AnswerList"][0]["Answer"]["AnswerText"]


def process_results_gen(doc, results):
    pred, refs = [results[0]], [doc_to_target(doc)]

    if len(refs[0]) < 1 or len(pred[0]) < 1:
        return {"rouge1": np.nan, "rouge2": np.nan, "rougeL": np.nan}

    rouge_results = rouge.compute(predictions=pred, references=refs)

    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
    }
