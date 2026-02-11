# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import re
import random
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests

BASE_URL = "http://judger:8000"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32
MODEL_NAME = "judge/model"

GENRM_PROMPT_TEMPLATE = """
You are an expert evaluator assessing AI model answers to questions using supporting documents.  
You will be provided with:
- A **Question**  
- A set of **Relevant Documents** (the gold standard grounding sources)  
- The **Correct Answer**  
- An **AI Model Solution**

Background:  
The AI model had access to a large pool of documents (indexed 0...N). Only a subset is truly relevant. Other documents may appear in citations but are simply distractors (not fabricated). The model's goal is to correctly answer the question while grounding its reasoning in the relevant documents.

Your task:  
Evaluate the model's solution objectively and consistently according to the criteria below. Do not use information outside the provided inputs.

---

[Question]  
{question}

[Relevant Documents]  
{gold_docs}

[Correct Answer]  
{answer}

[AI Model Solution]  
{solution}

---

### EVALUATION CRITERIA

**Criterion 1: Reasoning Quality (1 or 0)**  
Score **1** if the solution shows:  
- Clear logical flow from evidence to conclusion  
- No contradictions or fallacies  
- Coherent, well-structured reasoning  
Score **0** if the reasoning is flawed, contradictory, or incoherent.  

**Criterion 2: Document Grounding (1 or 0)**  
Score **1** if the solution:  
- Uses information primarily from the relevant documents  
- Represents those documents accurately (no distortions or false claims)  
- Does not rely significantly on irrelevant or external knowledge  
Score **0** if it misuses documents, ignores relevant evidence, or leans mainly on irrelevant/external sources.  

**Criterion 3: Answer Correctness (1 or 0)**  
Score **1** if the final answer matches the provided correct answer.  
Score **0** otherwise (including partial or incomplete answers).  

---

### RESPONSE FORMAT  
For each criterion, provide a 1-2 sentence justification followed by the score in the format below. Use only the exact box notation shown.  

Reasoning Quality Justification: [Your explanation]  
\\boxed{{Criterion 1: 1 or 0}}  

Document Grounding Justification: [Your explanation]  
\\boxed{{Criterion 2: 1 or 0}}  

Answer Correctness Justification: [Your explanation]  
\\boxed{{Criterion 3: 1 or 0}}  
"""


def get_response(solution_str, ground_truth, extra_info):
    question = extra_info["question"]
    gold_doc_ids = extra_info["relevant_gold_doc_ids"]
    gold_docs = extra_info["relevant_gold_docs"]
    gold_doc_string = ""
    for id, doc in zip(gold_doc_ids, gold_docs):
        gold_doc_string += f'{id}\n{doc}\n\n'

    prompt = GENRM_PROMPT_TEMPLATE.format(
        question=question,
        gold_docs=gold_doc_string,
        answer=ground_truth,
        solution=solution_str
    )

    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Content-Type": "application/json"}
            chat_url = f"{BASE_URL}/v1/chat/completions"
            data = {"model": MODEL_NAME, "messages": messages}
            output = requests.post(chat_url, headers=headers, json=data, timeout=240)
            response = output.json()["choices"][0]["message"]["content"]
            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    raise ConnectionRefusedError(f"Failed to run the model for {prompt}!")


def compute_reward(response):
    reward_score = 0.0
    
    # Pattern to match boxed criteria scores
    # Looks for: \boxed{Criterion X: 1/0} or \boxed{Criterion X: 1} or \boxed{Criterion X: 0}
    pattern = r'\\boxed\{\{?Criterion\s+\d+:\s*([01])(?:/[01])?\}?\}'
    
    # Find all matches
    matches = re.findall(pattern, response, re.IGNORECASE)
    
    if not matches:
        # Fallback: try simpler pattern without "Criterion" text
        fallback_pattern = r'\\boxed\{\{?([01])(?:/[01])?\}?\}'
        matches = re.findall(fallback_pattern, response, re.IGNORECASE)
        
        # Only use first 3 matches if we found any
        if matches:
            matches = matches[:3]
    
    # Sum up the scores
    for match in matches:
        try:
            score = int(match)
            if score in [0, 1]:  # Validate score is binary
                reward_score += float(score)
        except ValueError:
            continue
    
    # Ensure we don't exceed maximum possible score of 3.0
    reward_score = min(reward_score, 3.0)
    
    return reward_score


def compute_score(data_source, solution_str, ground_truth, extra_info):
    reward_score = 0.0
    response = get_response(solution_str, ground_truth, extra_info)
    if response is not None:
        reward_score = compute_reward(response)
    
    do_print = random.randint(1, 100) == 1
    if do_print:
        question = extra_info["question"]
        gold_doc_ids = extra_info["relevant_gold_doc_ids"]
        gold_docs = extra_info["relevant_gold_docs"]
        gold_doc_string = "\n"
        for id, doc in zip(gold_doc_ids, gold_docs):
            gold_doc_string += f'{id}\n{doc}\n\n'

        print("="*50)
        print("JUDGE REWARD FUNCTION DEBUGGING")
        print("-"*30)
        print(f'Question: {question}')
        print(f'Relevant documents: {gold_doc_string}')
        print(f'Golden answer: {ground_truth}')
        print("-"*30)
        print(f'Solution string: {solution_str}')
        print("-"*30)
        print(f'Judge response: {response}')
        print("-"*30)
        print(f'Reward: {reward_score}')
        print("="*50)

    return reward_score


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
            futures.append(future)

        results = [future.result() for future in futures]

    return results