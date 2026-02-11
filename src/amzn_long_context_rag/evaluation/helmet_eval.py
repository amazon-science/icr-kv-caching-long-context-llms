# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Adopted from https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/utils/eval_utils.py
"""
import re
import json
import string
import argparse
import jsonlines
import unicodedata
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from rouge_score import rouge_scorer

from loguru import logger

_DOC_ID_RE = re.compile(
    r"\[DOC\s+[^\]]+?\]",    # “[DOC ” + anything that is not “]” (lazy) + “]”
    re.IGNORECASE,
)

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_doc_ids(solution_str: str):
    """
    Return every document-ID token that matches the pattern “[DOC …]”.

    Duplicates are removed while preserving the first-appearing order.
    """
    raw_ids = _DOC_ID_RE.findall(solution_str)
    seen = set()
    unique_ids = []
    for doc_id in raw_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique_ids.append(doc_id)
    return unique_ids


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def compute_doc_ids_f1(solution_str, gold_doc_ids):
    """
    Version 1: Use F1 score for document ID matching
    Balances precision and recall to discourage outputting all IDs
    """
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    extracted_doc_ids = set(extract_doc_ids(solution_str=solution_str))
    gold_doc_ids = set(gold_doc_ids)

    # Document ID matching with F1 score
    if len(extracted_doc_ids) > 0 and len(gold_doc_ids) > 0:
        correct = extracted_doc_ids & gold_doc_ids
        precision = len(correct) / len(extracted_doc_ids)
        recall = len(correct) / len(gold_doc_ids)
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score
    }

    return metrics


def drqa_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def drqa_exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediciton, ground_truth):
    """Check if the ground truth is a (soft) exact match substring of the prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediciton) 


def drqa_metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    # ground truth could be a string or a list of strings or a list of list of strings
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif isinstance(ground_truths[0], list):
        ground_truths = [ground_truth for ground_truths_list in ground_truths for ground_truth in ground_truths_list]

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def parse_output(output, prefix="Answer:"):
    def lstrip_string(s, sub):
        return re.sub(f'^{re.escape(sub)}', '', s, flags=re.IGNORECASE)
    patterns = [
        re.compile(r"(?:\b(?:the\s+)?answer(?:\s+\w+)*\s*[:\-]?\s*)([^\.,\n]+)", flags=re.IGNORECASE), ## Added to comply with fine-tuned model outputs.
        re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), 
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pat in patterns:
        matches = pat.search(output)
        if matches is not None:
            return lstrip_string(matches[1].strip(), prefix).strip() # 0 index includes the non-capturing group # lstrip again because for chat models sometimes it will repeat the prefix
    # if still not found, return None, but should actually never get this case...
    return None



def calculate_metrics(prediction, answers):
    r_scorer = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
    em = drqa_metric_max_over_ground_truths(drqa_exact_match_score, prediction, answers)
    f1 = drqa_metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], prediction, answers)
    sub_em = drqa_metric_max_over_ground_truths(substring_exact_match_score, prediction, answers)

    if isinstance(answers, str):
        answers = [answers]
    elif isinstance(answers[0], list):
        answers = [ground_truth for ground_truths_list in answers for ground_truth in ground_truths_list]

    rouges = [r_scorer.score(target=a, prediction=prediction) for a in answers]
    rouge = {}
    for k in r_scorer.rouge_types:
        rouge[k + "_f1"] = max([r[k].fmeasure for r in rouges])
        rouge[k + "_recall"] = max([r[k].recall for r in rouges])

    return {
        "exact_match": em,
        "f1": f1,
        "substring_exact_match": sub_em,
        **rouge,
    }


def default_post_process(output, example):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    prediction = output["output"]
    answer = example["answer"]
    mets = calculate_metrics(prediction, answer)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(prediction)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    return mets, {"parsed_output": parsed_pred}


def compute_scores(task: str, file_path: Path, output_dir: Path):

    out_subdir = (
        Path(output_dir).expanduser()
        / Path("/".join(file_path.parts[2:-1]))
    )
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{file_path.stem}.json"

    with jsonlines.open(file_path, "r") as fin:
        lines = [line for line in fin]

    metrics = defaultdict(list)

    # TODO: implement evalution for other HELMET subsets (using the 'task' variable).
    if "kilt" in task:
        for line in tqdm(lines, total=len(lines)):
            output = {"output": line["model_output"]}
            test_item = {"answer": line["answers"]}

            mets, _ = default_post_process(output, test_item)
            for k, v in mets.items():
                metrics[k].append(v)

            if "gold_doc_ids" in line:
                solution_str = line["model_output"]
                gold_doc_ids = line["gold_doc_ids"]
                doc_ids_metrics = compute_doc_ids_f1(solution_str, gold_doc_ids)
                metrics["doc_ids_precision"].append(doc_ids_metrics["precision"])
                metrics["doc_ids_recall"].append(doc_ids_metrics["recall"])
                metrics["doc_ids_f1"].append(doc_ids_metrics["f1"])

        averaged_metrics = {k: np.mean(v)*(100) for k, v in metrics.items()}

        logger.info("Averaged metrics:")
        for k, v in averaged_metrics.items():
            logger.info(f"{k}: {v:.02f}")

        with open(out_path, "w") as f:
            json.dump(averaged_metrics, f, indent=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default='kilt', help="Which task to evaluate`.")  # noqa
    p.add_argument('--file', type=Path, required=True, help="JSONL result file")
    p.add_argument("--output_dir", type=Path, default="data/evaluation", help="Where to write the prediction results.")  # noqa
    args = p.parse_args()
    compute_scores(args.task, args.file, args.output_dir)