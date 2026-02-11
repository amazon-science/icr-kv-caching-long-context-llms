# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random
import re
import string
from typing import List, Set, Tuple, Dict
from collections import Counter


_DOC_ID_RE = re.compile(
    r"\[DOC\s+[^\]]+?\]",    # "[DOC " + anything that is not "]" (lazy) + "]"
    re.IGNORECASE,
)

def normalize_answer(s):
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


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_doc_ids(solution_str: str):
    """
    Return every document-ID token that matches the pattern "[DOC …]".
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


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r"(?i)The answer is:\s*([^\n\.]+)"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def char_level_f1(text1: str, text2: str) -> float:
    """Calculate character-level F1 score."""
    chars1 = Counter(text1.lower().replace(" ", ""))
    chars2 = Counter(text2.lower().replace(" ", ""))
    
    common = sum((chars1 & chars2).values())
    total1 = sum(chars1.values())
    total2 = sum(chars2.values())
    
    if total1 == 0 and total2 == 0:
        return 1.0
    if total1 == 0 or total2 == 0:
        return 0.0
    
    precision = common / total1
    recall = common / total2
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def ngram_jaccard(text1: str, text2: str, n: int) -> float:
    """Calculate n-gram Jaccard similarity."""
    def get_ngrams(text: str, n: int) -> Set[str]:
        text = text.lower().replace(" ", "")
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0


def word_jaccard(text1: str, text2: str) -> float:
    """Calculate word-level Jaccard similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def fuzzy_match_score(text1: str, text2: str, threshold: float = 0.8) -> float:
    """
    Combine multiple fuzzy matching methods and return a continuous score.
    Returns the actual similarity score (0.0 to 1.0).
    """
    # Character-level F1
    char_f1 = char_level_f1(text1, text2)
    
    # N-gram Jaccard (using bigrams and trigrams)
    bigram_jaccard = ngram_jaccard(text1, text2, n=2)
    trigram_jaccard = ngram_jaccard(text1, text2, n=3)
    
    # Word-level Jaccard
    word_jacc = word_jaccard(text1, text2)
    
    # Take the maximum of all methods
    max_score = max(char_f1, bigram_jaccard, trigram_jaccard, word_jacc)
    
    return max_score


_QUOTE_RE = re.compile(
    r"(?:Quote|Passage|Document|Extract|Reference)\s+\d+:\s*(.*?)(?=(?:Quote|Passage|Document|Extract|Reference)\s+\d+:|Relevant Document IDs?:|The answer is:|$)",
    re.IGNORECASE | re.DOTALL
)

def extract_quotes(solution_str: str) -> List[str]:
    """
    Extract quotes from solution string using simple regex pattern.
    Captures everything after "Quote X:" until the next quote or end of string.
    """
    quotes = []
    
    # Find all quotes using the regex
    matches = _QUOTE_RE.findall(solution_str)
    
    for match in matches:
        quote_content = match.strip()
        # Filter out very short quotes
        if len(quote_content) > 5:
            quotes.append(quote_content)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_quotes = []
    for quote in quotes:
        if quote not in seen:
            seen.add(quote)
            unique_quotes.append(quote)
    
    return unique_quotes[:10]  # Limit to max 10 quotes


def check_format_compliance(solution_str: str, quotes: List[str], max_quotes: int) -> Tuple[float, dict]:
    """
    Comprehensive format checking function.
    Returns format score (0.0 to 1.0) and detailed breakdown.
    """
    format_breakdown = {
        'has_quotes_section': False,
        'has_doc_ids_section': False,
        'has_answer_section': False,
        'max_quotes': False,
    }
    
    # Check for quotes section and proper formatting
    quote_pattern = r'(?:Quote|Passage|Document|Extract|Reference)\s+\d+:\s*["\'][^"\']+["\']'
    if re.search(quote_pattern, solution_str, re.IGNORECASE):
        format_breakdown['has_quotes_section'] = True
    
    # Check for document IDs section
    doc_ids_header_pattern = r'Relevant Document IDs?:'
    if re.search(doc_ids_header_pattern, solution_str, re.IGNORECASE):
        format_breakdown['has_doc_ids_section'] = True
    
    # Check for answer section
    answer_pattern = r'The answer is:\s*'
    if re.search(answer_pattern, solution_str, re.IGNORECASE):
        format_breakdown['has_answer_section'] = True
    
    # Check max quote count (1-10)
    if 1 <= len(quotes) <= max_quotes:
        format_breakdown['max_quotes'] = True
        
    # Calculate format score (weighted)
    weights = {
        'has_quotes_section': 0.25,
        'has_doc_ids_section': 0.25,
        'has_answer_section': 0.25,
        'max_quotes': 0.25,
    }
    
    format_score = sum(weights[key] for key, value in format_breakdown.items() if value)
    
    return format_score


def bipartite_matching(quotes: List[str], gold_docs: List[str], threshold: float = 0.8) -> Tuple[List[Tuple[int, int, float]], int, int]:
    """
    Perform bipartite matching between quotes and gold docs.
    Each quote can match at most one gold doc, and vice versa.
    
    Returns:
        - List of (quote_idx, gold_doc_idx, similarity_score) for successful matches
        - Number of verified quotes
        - Number of covered gold docs
    """
    if not quotes or not gold_docs:
        return [], 0, 0
    
    # Calculate all pairwise similarities
    similarities = []
    for i, quote in enumerate(quotes):
        for j, gold_doc in enumerate(gold_docs):
            score = fuzzy_match_score(quote, gold_doc, threshold)
            if score >= threshold:
                similarities.append((i, j, score))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy matching: assign highest scoring pairs first
    matched_quotes = set()
    matched_gold_docs = set()
    successful_matches = []
    
    for quote_idx, gold_doc_idx, score in similarities:
        if quote_idx not in matched_quotes and gold_doc_idx not in matched_gold_docs:
            successful_matches.append((quote_idx, gold_doc_idx, score))
            matched_quotes.add(quote_idx)
            matched_gold_docs.add(gold_doc_idx)
    
    return successful_matches, len(matched_quotes), len(matched_gold_docs)


def count_tokens(text: str) -> int:
    """Simple token counting by splitting on whitespace."""
    return len(text.split())


def my_reward_fn_quote(data_source, solution_str, ground_truth, extra_info=None):
    """
    Quote reward function with 6 components:
    R_quote + R_coverage + R_id + R_answer + R_format - R_len_penalty
    """
    # Weights for each component (based on importance)
    w_quote = 1.0      # Precision at string level - important for learning relevance
    w_coverage = 1.0   # Recall at string level - important for completeness  
    w_id = 1.0         # Document ID accuracy - helps with structure
    w_answer = 1.0     # Final answer correctness - most important
    w_format = 0.2     # Format compliance - small bonus
    w_len_penalty = 0.1 # Length penalty - prevents overly long quotes
    
    # Extract components from solution
    quotes = extract_quotes(solution_str)
    doc_ids = set(extract_doc_ids(solution_str))
    answer = extract_solution(solution_str)
    
    # Get gold data
    gold_docs = extra_info.get("relevant_gold_docs", [])
    gold_doc_ids = set(extra_info.get("relevant_gold_doc_ids", []))
    
    # Initialize rewards
    r_quote = 0.0
    r_coverage = 0.0
    r_id = 0.0
    r_answer = 0.0
    r_format = 0.0
    r_len_penalty = 0.0
    
    # R_answer: Binary sub-exact-match (most important)
    if answer is not None and subem_check(answer, ground_truth):
        r_answer = 1.0
    
    # R_quote & R_coverage: Use bipartite matching to prevent double-counting
    successful_matches, num_verified_quotes, num_covered_docs = bipartite_matching(quotes, gold_docs, threshold=0.8)
    
    # R_quote: Precision at string level
    r_quote = num_verified_quotes / len(quotes) if quotes else 0.0
    
    # R_coverage: Recall at string level  
    r_coverage = num_covered_docs / len(gold_docs) if gold_docs else 0.0
    
    # R_id: F1 on document IDs, only when we have verified quotes
    # This prevents reward hacking by requiring actual quote matching first
    if num_verified_quotes > 0 and gold_doc_ids and doc_ids:
        correct_ids = doc_ids & gold_doc_ids
        precision = len(correct_ids) / len(doc_ids)
        recall = len(correct_ids) / len(gold_doc_ids)
        if precision + recall > 0:
            r_id = 2 * precision * recall / (precision + recall)
    
    # R_format: Comprehensive format compliance check
    r_format = check_format_compliance(solution_str, quotes, max_quotes=10)
    
    # R_len_penalty: Penalty for quotes >30 tokens
    penalty = 0.0
    for quote in quotes:
        tokens = count_tokens(quote)
        if tokens > 30:
            # Progressive penalty: more severe for longer quotes
            penalty += 0.1 + 0.02 * (tokens - 30)
    r_len_penalty = min(penalty, 1.0)  # Cap penalty at 1.0
    
    # Total reward
    total_reward = (w_quote * r_quote + 
                   w_coverage * r_coverage + 
                   w_id * r_id + 
                   w_answer * r_answer + 
                   w_format * r_format - 
                   w_len_penalty * r_len_penalty)
    
    # Debugging (occasional printing)
    do_print = random.randint(1, 100) == 1
    if do_print:
        print("="*50)
        print("QUOTE REWARD FUNCTION DEBUGGING")
        print("-" * 30)
        print(f"Solution string: {solution_str}")
        print("-" * 30)
        print(f"Extracted quotes ({len(quotes)}): {quotes}")
        print(f"Gold docs ({len(gold_docs)}): {gold_docs}")
        print(f"Successful matches: {len(successful_matches)}")
        for i, (quote_idx, gold_idx, score) in enumerate(successful_matches):
            print(f"  Match {i+1}: Quote[{quote_idx}] <-> Gold[{gold_idx}] (score: {score:.3f})")
        print(f"Verified quotes: {num_verified_quotes}/{len(quotes)}")
        print(f"Covered gold docs: {num_covered_docs}/{len(gold_docs)}")
        print("-" * 30)
        print(f"Extracted doc IDs: {doc_ids}")
        print(f"Gold doc IDs: {gold_doc_ids}")
        print("-" * 30)
        print(f"Extracted answer: {answer}")
        print(f"Golden answer: {ground_truth}")
        print("-" * 30)
        print(f"R_quote (precision): {r_quote:.3f} (weight: {w_quote})")
        print(f"R_coverage (recall): {r_coverage:.3f} (weight: {w_coverage})")
        print(f"R_id (F1): {r_id:.3f} (weight: {w_id})")
        print(f"R_answer: {r_answer:.3f} (weight: {w_answer})")
        print(f"R_format: {r_format:.3f} (weight: {w_format})")
        print(f"R_len_penalty: {r_len_penalty:.3f} (weight: {w_len_penalty})")
        print(f"Total reward: {total_reward:.3f}")
        print("="*50)
    
    return max(0.0, total_reward)  # Ensure non-negative


def extract_full_doc_strings(solution_str: str) -> List[str]:
    """
    Extract complete [DOC i] content strings from the solution.
    Handles unquoted format: [DOC i] content text
    """
    doc_strings = []
    
    # Pattern: [DOC i] content (until next [DOC or end)
    pattern = r'\[DOC\s+(\d+)\]\s+([^\[]+?)(?=\s*\[DOC|\s*The answer is:|$)'
    matches = re.findall(pattern, solution_str, re.IGNORECASE | re.DOTALL)
    
    for doc_num, content in matches:
        content_clean = content.strip()
        if content_clean:
            doc_string = f'[DOC {doc_num}] {content_clean}'
            doc_strings.append(doc_string)
    
    return doc_strings


def parse_relevant_documents_section(solution_str: str) -> List[Tuple[str, str]]:
    """
    Parse the 'Relevant documents:' section to extract document IDs and their content.
    Returns list of (doc_id, content) tuples.
    """
    # Find the relevant documents section
    relevant_docs_pattern = r"Relevant documents:\s*(.*?)(?=The answer is:|$)"
    match = re.search(relevant_docs_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    
    if not match:
        return []
    
    relevant_section = match.group(1).strip()
    
    # Extract individual documents with their content
    # Pattern: [DOC X] followed by content until next [DOC Y] or end
    doc_pattern = r"\[DOC\s+(\d+)\]\s*(.*?)(?=\[DOC\s+\d+\]|$)"
    
    extracted_docs = []
    for match in re.finditer(doc_pattern, relevant_section, re.DOTALL):
        doc_id = f"[DOC {match.group(1)}]"
        content = match.group(2).strip()
        if content:  # Only add if there's actual content
            extracted_docs.append((doc_id, content))
    
    return extracted_docs


def check_document_format_compliance(solution_str: str) -> Tuple[float, Dict[str, bool]]:
    """
    Check if the solution follows the expected format.
    """
    format_checks = {
        'has_relevant_docs_header': False,
        'has_doc_ids_in_relevant_section': False, 
        'has_answer_section': False,
        'proper_document_structure': False
    }
    
    # Check for "Relevant documents:" header
    if re.search(r"Relevant documents:\s*", solution_str, re.IGNORECASE):
        format_checks['has_relevant_docs_header'] = True
    
    # Check for document IDs in relevant section
    relevant_docs_pattern = r"Relevant documents:\s*(.*?)(?=The answer is:|$)"
    match = re.search(relevant_docs_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if match and re.search(r"\[DOC\s+\d+\]", match.group(1)):
        format_checks['has_doc_ids_in_relevant_section'] = True
        
        # Check if documents have content (not just IDs)
        if re.search(r"\[DOC\s+\d+\]\s*\S", match.group(1)):
            format_checks['proper_document_structure'] = True
    
    # Check for answer section
    if re.search(r"The answer is:\s*", solution_str, re.IGNORECASE):
        format_checks['has_answer_section'] = True
    
    # Calculate format score
    format_score = sum(format_checks.values()) / len(format_checks)
    
    return format_score, format_checks


def my_reward_fn_doc_ids_and_content(data_source, solution_str, ground_truth, extra_info=None):
    """
    Comprehensive reward function for document selection task.
    
    Components:
    1. R_doc_id: F1 score for document ID matching (prevents outputting all IDs)
    2. R_content: Content accuracy for matched documents
    3. R_answer: Final answer correctness  
    4. R_format: Format compliance
    5. R_penalty: Penalty for including irrelevant documents
    """
    
    # Component weights
    w_doc_id = 1.5      # Document ID accuracy (important for learning relevance)
    w_content = 1.0     # Content reproduction accuracy
    w_answer = 2.0      # Final answer (most important)
    w_format = 0.3      # Format compliance
    w_penalty = 1.0     # Penalty for irrelevant documents
    
    # Initialize rewards
    r_doc_id = 0.0
    r_content = 0.0
    r_answer = 0.0
    r_format = 0.0
    r_penalty = 0.0
    
    # Extract components from solution
    extracted_docs = parse_relevant_documents_section(solution_str)
    extracted_doc_ids = set([doc_id for doc_id, _ in extracted_docs])
    answer = extract_solution(solution_str)
    
    # Get gold data
    gold_doc_ids = set(extra_info.get("relevant_gold_doc_ids", []))
    irrelevant_docs = extracted_doc_ids - gold_doc_ids  # Calculate here for use in penalty and debugging
    gold_docs_dict = {}
    
    # Create mapping from gold doc IDs to content
    if "relevant_gold_docs" in extra_info:
        gold_docs = extra_info["relevant_gold_docs"]
        # Assume gold docs are in same order as gold doc IDs
        for i, doc_id in enumerate(extra_info.get("relevant_gold_doc_ids", [])):
            if i < len(gold_docs):
                gold_docs_dict[doc_id] = gold_docs[i]
    
    # R_answer: Answer correctness (most important)
    if answer is not None and subem_check(answer, ground_truth):
        r_answer = 1.0
    
    # R_doc_id: F1 score for document ID matching
    if extracted_doc_ids and gold_doc_ids:
        correct_ids = extracted_doc_ids & gold_doc_ids
        precision = len(correct_ids) / len(extracted_doc_ids)
        recall = len(correct_ids) / len(gold_doc_ids)
        
        if precision + recall > 0:
            r_doc_id = 2 * precision * recall / (precision + recall)
    
    # R_content: Content accuracy for correctly identified documents
    content_scores = []
    correctly_identified_docs = extracted_doc_ids & gold_doc_ids
    
    for doc_id in correctly_identified_docs:
        # Find extracted content for this doc ID
        extracted_content = None
        for ext_id, ext_content in extracted_docs:
            if ext_id == doc_id:
                extracted_content = ext_content
                break
        
        if extracted_content and doc_id in gold_docs_dict:
            gold_content = gold_docs_dict[doc_id]
            similarity = fuzzy_match_score(extracted_content, gold_content)
            content_scores.append(similarity)
    
    r_content = sum(content_scores) / len(content_scores) if content_scores else 0.0
    
    # R_format: Format compliance
    r_format, _ = check_document_format_compliance(solution_str)
    
    # R_penalty: Penalty for including irrelevant documents
    r_penalty = len(irrelevant_docs) * 0.3  # 0.3 penalty per irrelevant doc
    
    # Total reward
    total_reward = (w_doc_id * r_doc_id + 
                   w_content * r_content + 
                   w_answer * r_answer + 
                   w_format * r_format - 
                   w_penalty * r_penalty)
    
    # Debugging (occasional printing)
    do_print = random.randint(1, 100) == 1
    if do_print:
        print("="*60)
        print("DOC IDS AND CONTENT REWARD DEBUGGING")
        print("-" * 40)
        print(f"Solution string: {solution_str}...")
        print("-" * 40)
        print(f"Extracted docs: {len(extracted_docs)}")
        for i, (doc_id, content) in enumerate(extracted_docs):
            print(f"  {doc_id}: {content}")
        print(f"Extracted doc IDs: {extracted_doc_ids}")
        print(f"Gold doc IDs: {gold_doc_ids}")
        print(f"Correctly identified: {extracted_doc_ids & gold_doc_ids}")
        print(f"Irrelevant docs: {irrelevant_docs}")
        print("-" * 40)
        print(f"Extracted answer: {answer}")
        print(f"Gold answer: {ground_truth}")
        print("-" * 40)
        print(f"R_doc_id (F1): {r_doc_id:.3f} (weight: {w_doc_id})")
        print(f"R_content: {r_content:.3f} (weight: {w_content})")
        print(f"R_answer: {r_answer:.3f} (weight: {w_answer})")
        print(f"R_format: {r_format:.3f} (weight: {w_format})")
        print(f"R_penalty: {r_penalty:.3f} (weight: {w_penalty}) [irrelevant docs: {len(irrelevant_docs)}]")
        print(f"Total reward: {total_reward:.3f}")
        print("="*60)
    
    return max(0.0, total_reward)


def my_reward_fn_answer_only(data_source, solution_str, ground_truth, extra_info=None):
    """
    Use SubEM for answer only matching.
    """
    score = 0.0
    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth):
            score += 1.0
    
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("||||||||||||||||||||||||||||||||")
        print("REWARD FUNCTION DEBUGGING (ANSWER ONLY)")
        print("--------------------------------")
        print(f'Solution string: {solution_str}')
        print("--------------------------------")
        print(f'Extracted answer: {answer}')
        print(f'Golden answer: {ground_truth}')
        print("--------------------------------")

    return score

def my_reward_fn_f1_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Version 1: Use F1 score for document ID matching
    Balances precision and recall to discourage outputting all IDs
    """
    score = 0.0
    answer = extract_solution(solution_str=solution_str)
    extracted_doc_ids = set(extract_doc_ids(solution_str=solution_str))
    gold_doc_ids = set(extra_info["gold_doc_ids"])

    if answer is None:
        return 0
    else:
        # Answer correctness
        if subem_check(answer, ground_truth):
            score += 1.0
        
        # Document ID matching with F1 score
        if len(extracted_doc_ids) > 0 and len(gold_doc_ids) > 0:
            correct = extracted_doc_ids & gold_doc_ids
            precision = len(correct) / len(extracted_doc_ids)
            recall = len(correct) / len(gold_doc_ids)
            
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
                score += f1_score
    
    do_print = random.randint(1, 32) == 1
    if do_print:
        print("||||||||||||||||||||||||||||||||")
        print("REWARD FUNCTION DEBUGGING (F1)")
        print("--------------------------------")
        print(f'Solution string: {solution_str}')
        print("--------------------------------")
        print(f'Extracted doc ids: {extracted_doc_ids}')
        print(f'Gold doc ids: {extra_info["gold_doc_ids"]}')
        print("--------------------------------")
        print(f'Extracted answer: {answer}')
        print(f'Golden answer: {ground_truth}')
        print("--------------------------------")
        if len(extracted_doc_ids) > 0 and len(gold_doc_ids) > 0:
            correct = extracted_doc_ids & gold_doc_ids
            precision = len(correct) / len(extracted_doc_ids)
            recall = len(correct) / len(gold_doc_ids)
            print(f'Precision: {precision:.3f}, Recall: {recall:.3f}')
        print(f'Reward: {score}')

    return score