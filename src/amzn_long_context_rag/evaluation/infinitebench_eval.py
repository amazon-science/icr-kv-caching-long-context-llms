# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from pathlib import Path
import json
import jsonlines
import re
import string
from collections import Counter

from tqdm import tqdm
import evaluate
import argparse


ROUGE_SCORER = evaluate.load("rouge")

def _extract_answer(text: str) -> str | None:
    # pattern = re.compile(r"The answer is\s+((?:[A-Za-z]+\.\s+)*[^.]+?)\.", re.I)
    pattern = re.compile(r"The answer is\s*:?\s*((?:[A-Za-z]+\.\s+)*[^.,]+?)[.,]", re.I) ## Slightly modified to accomodate fine-tuned model outputs.
    match = pattern.search(text)
    return match.group(1).strip() if match else text

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Chinese version. Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."  # noqa
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth) -> tuple[float, float, float]:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths) -> float:
    """Computes the F1, recall, and precision."""
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        extracted_pred = _extract_answer(pred)
        normalized_prediction = normalize_answer(extracted_pred)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def qa_f1_score_zh(pred: str, ground_truths: list[str]) -> float:
    """
    QA F1 score for chinese.
    """
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        norm_pred = normalize_zh_answer(pred)
        norm_label = normalize_zh_answer(ground_truth)

        # One character one token.
        pred_tokens = list(norm_pred)
        label_tokens = list(norm_label)
        scores = f1_score(pred_tokens, label_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def load_json(fname):
    return json.load(open(fname))


def iter_jsonl(fname, maximum_input_length):
    lines = []
    with jsonlines.open(fname, "r") as fin:
        for line in fin:
            original_ctx_length = line["original_ctx_length"]

            if isinstance(maximum_input_length, int):
                if original_ctx_length > maximum_input_length:
                    continue
            
            if isinstance(maximum_input_length, list):
                if original_ctx_length < maximum_input_length[0] or original_ctx_length > maximum_input_length[1]:
                    continue

            lines.append(line)
    return lines

def first_int_match(prediction):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def split_retrieval_answer(pred: str):
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    return words


def get_score_one_kv_retrieval(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    for c in ['\n', ':', '\"', '\'', '.', ',', '?', '!', '{', '}']:
        pred = pred.replace(c, ' ')
    words = pred.split()
    return label in words


def get_score_one_passkey(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_number_string(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_code_run(pred, label) -> bool:
    """
    Returns the score of one example in Code.Run.
    """
    if isinstance(label, list):
        label = label[0]
        if isinstance(label, str):
            label = int(label)
    pred = pred.strip()
    for c in ["\n", ".", "`", "'", '"', ":"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    if len(words) == 0:
        return False
    try:
        pred = int(words[-1])
        return label == pred
    except Exception:
        return False


def get_score_one_code_debug(pred, label) -> bool:
    """
    Returns the score of one example in Code.Debug.
    """
    pred = pred.strip()
    label_c = label[1]
    fn_name = label[0]
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred == label_c:
            return True
    ans_prefixes = [
        "answer is:",
        # "answer is",
        # "error is",
        "is:",
        "answer:",
        "correct option is:"
    ]
    pred = pred.strip()
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    if pred.startswith(label_c) or pred.startswith(fn_name):
        return True
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        pred = pred[idx + len(prefix) + 1 :]
        for s in [label_c, fn_name]:
            if pred.startswith(s):
                return True
        return False
    return False


def get_score_one_math_find(pred, label) -> bool:
    if isinstance(label, list):
        # In math_find, there is always only one label.
        label = label[0]
        if isinstance(label, str):
            label = int(label)
    if isinstance(label, int):
        # Find first int or float
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return False
        first_num = first_num.group(0).strip()
        return int(first_num) == label
    elif isinstance(label, float):
        # Find first float or int
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return False
        first_float = first_float.group(0).strip()
        return float(first_float) == label
    else:
        raise TypeError(f"Expected int or float, got {type(label)}")


def get_score_one_longdialogue_qa_eng(pred, label) -> bool:
    pred = pred.strip()
    pred = pred.upper()
    for item in label:
        if item.upper() in pred:
            return 1
    return 0


def get_score_one_longbook_choice_eng(pred, label) -> bool:
    # Just use the first letter as the prediction
    pred = pred.strip()
    pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"

    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred in label:
            return True
    if pred == "":
        return False
    if pred[0] in "ABCD":
        return pred[0] in label
    if pred in label:
        return True
    # Find a answer prefix
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "answer is ",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally, just find the first occurrence of A, B, C, or D.
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False


# Original logic implemented by InfiniteBench
# def get_score_one_longbook_qa_eng(pred, label) -> float:
#     return qa_f1_score(pred, label)


# Substitute with SubEM like in HELMET
def get_score_one_longbook_qa_eng(pred, label) -> float:
    # Original logic implemented by InfiniteBench
    f1 = qa_f1_score(pred, label)

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
    
    # We keep only SubEM
    sub_em = drqa_metric_max_over_ground_truths(substring_exact_match_score, pred, label)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(pred)
    if parsed_pred is not None:
        new_sub_em = drqa_metric_max_over_ground_truths(substring_exact_match_score, parsed_pred, label)
        sub_em = max(sub_em, new_sub_em)

    return {"f1": f1, "sub_em": sub_em}
        

def get_score_one_longbook_sum_eng(
    pred: str, label: str
) -> float:

    score = ROUGE_SCORER.compute(
        predictions=[pred], references=[label], use_aggregator=False
    )
    return score["rougeLsum"][0]  # type: ignore


def get_score_one_longbook_qa_chn(pred, label) -> float:
    return qa_f1_score_zh(pred, label)


def get_score_one_math_calc(pred, label) -> float:
    assert isinstance(label, list), f"Expected list, got {type(label)}"
    # assert isinstance(pred, list), f"Expected list, got {type(pred)}"
    if isinstance(label[0], list):
        label = label[0]
    pred_nums = []
    pred_list = re.split("[^0-9]", pred)
    for item in pred_list:
        if item != "":
            pred_nums.append(int(item))

    # Our prompts makes GPT4 always output the first number as the first value
    # in the predicted answer.
    # if model_name == "gpt4":
    #     pred_nums = pred_nums[1:]

    cnt = 0
    for i in range(len(label)):
        if i >= len(pred_nums):
            break
        if label[i] == pred_nums[i]:
            cnt += 1
        else:
            break
    return cnt / len(label)


def get_score_one(
    pred: str, label: str, task_name: str
) -> float:
    """
    Computes the score for one prediction.
    Returns one float (zero and one for boolean values).
    """
    NAME_TO_SCORE_GETTER = {
        # Retrieve
        "kv_retrieval": get_score_one_kv_retrieval,
        "kv_retrieval_prefix": get_score_one_kv_retrieval,
        "kv_retrieval_both": get_score_one_kv_retrieval,

        "passkey": get_score_one_passkey,
        "number_string": get_score_one_number_string,
        # Code
        "code_run": get_score_one_code_run,
        "code_debug": get_score_one_code_debug,
        # Longbook
        "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
        "longbook_qa_eng": get_score_one_longbook_qa_eng,
        "longbook_sum_eng": get_score_one_longbook_sum_eng,
        "longbook_choice_eng": get_score_one_longbook_choice_eng,
        "longbook_qa_chn": get_score_one_longbook_qa_chn,
        # Math
        "math_find": get_score_one_math_find,
        "math_calc": get_score_one_math_calc,
    }
    assert task_name in NAME_TO_SCORE_GETTER, f"Invalid task name: {task_name}"
    score = NAME_TO_SCORE_GETTER[task_name](pred, label)
    return score


def get_labels(preds: list) -> list[str]:
    possible_label_keys = ["ground_truth", "label", "answer"]
    for label_key in possible_label_keys:
        if label_key in preds[0]:
            return [x.get(label_key, "XXXXXXXXXX") for x in preds]
    raise ValueError(f"Cannot find label in {preds[0]}")


def get_preds(preds: list) -> list[str]:
    pred_strings = []
    possible_pred_keys = ["prediction", "pred", "model_output"]
    for pred in preds:
        this_pred = "NO PREDICTION"
        for pred_key in possible_pred_keys:
            if pred_key in pred:
                this_pred = pred[pred_key]
                break
        else:
            raise ValueError(f"Cannot find prediction in {pred}")
        pred_strings.append(this_pred)
    return pred_strings


def get_score(
    labels: list, preds: list, task_name: str
) -> float:
    """
    Computes the average score for a task.
    """
    assert len(labels) == len(preds)
    ## Dictionary containing a generic "score" (for rouge, etc.) and two specific keys for 
    ## the QA task: f1 and sub_em
    scores = {"score": [], "f1": [], "sub_em": []}
    for label, pred in tqdm(zip(labels, preds)):
        score = get_score_one(pred, label, task_name)
        ## handle normal case with a single score
        if isinstance(score, float) or isinstance(score, int):
            scores["score"].append(score)
        ## handle f1 and sub_em for the QA task
        if isinstance(score, dict):
            scores["f1"].append(score["f1"])
            scores["sub_em"].append(score["sub_em"])

    scores = {k: v for k, v in scores.items() if v != []}
    for k, v in scores.items():
        avg = sum(v) / len(v)
        scores[k] = avg
    return scores


def compute_scores(preds_path: Path, out_path: Path, task: str, maximum_input_length: list):
    print("Loading prediction results from", preds_path)
    preds = iter_jsonl(preds_path, maximum_input_length)
    labels = get_labels(preds)
    preds = get_preds(preds)

    acc = get_score(labels, preds, task)

    print(f'{task} evaluation results = {acc}')

    with open(out_path, "w") as fout:
        if isinstance(acc, float) or isinstance(acc, int):
            json.dump(
                {
                    "task": task, 
                    "score": acc,
                    "instances": len(preds),
                    "maximum_input_length": maximum_input_length
                }, 
                fout, 
                indent=2
            )
        if isinstance(acc, dict):
            json.dump(
                {
                    "task": task, 
                    **acc,
                    "instances": len(preds),
                    "maximum_input_length": maximum_input_length
                }, 
                fout, 
                indent=2
            )


ALL_TASKS = [
    "passkey",
    "number_string",
    "kv_retrieval",
    "longdialogue_qa_eng",
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "longbook_qa_chn",
    "math_find",
    "math_calc",
    "code_run",
    "code_debug",
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.")  # noqa
    p.add_argument('--file', type=Path, required=True, help="JSONL result file")
    p.add_argument('--maximum_input_length', type=int, nargs='+', help="The maximum number of context tokens. Used to filer out instances.")
    p.add_argument("--output_dir", type=Path, default="data/evaluation", help="Where to write the prediction results.")  # noqa
    args = p.parse_args()
    if args.task == "all":
        tasks = ALL_TASKS
    else:
        tasks = [args.task]
    for task in tasks:
        out_subdir = (
            Path(args.output_dir).expanduser()
            / Path("/".join(args.file.parts[2:-1]))
        )
        out_subdir.mkdir(parents=True, exist_ok=True)

        maximum_input_length = args.maximum_input_length
        if maximum_input_length and len(maximum_input_length) == 1:
            maximum_input_length = maximum_input_length[0]

        if maximum_input_length:
            if isinstance(maximum_input_length, int):
                out_path = out_subdir / f"{args.file.stem}_maximum_input_length={maximum_input_length}.json"
            if isinstance(maximum_input_length, list):
                out_path = out_subdir / f"{args.file.stem}_maximum_input_length=[{maximum_input_length[0]},{maximum_input_length[1]}].json"    
        else:
            out_path = out_subdir / f"{args.file.stem}.json"
        
        compute_scores(args.file, out_path, task, maximum_input_length)