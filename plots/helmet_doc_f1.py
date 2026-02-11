# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import re
import json
import matplotlib.pyplot as plt
from collections import defaultdict

"""
Scans directories like:
  data/evaluation/HELMET/kilt/<benchmark>-dev-multikilt_1000_k<num_retrieved>_dep<num_depth>/
Within each of those, there are per-model subfolders that contain zero_shot_doc_ids.json.
Plots doc_ids_f1 vs context length with one line per model, averaging 440/500 into 64k.
"""

# Base directory containing the benchmark-run folders
BASE_DIR = "data/evaluation/HELMET/kilt"
OUT_DIR="plots"

# Map retrieved-doc counts to display context labels
CONTEXT_MAP = {
    "20": "4k",
    "50": "8k",
    "105": "16k",
    "220": "32k",
    "440": "64k",
    "1000": "128k",
}

# Fixed x-axis order
CONTEXT_ORDER = ["4k", "8k", "16k", "32k", "64k", "128k"]

# Folder name pattern, e.g. "hotpotqa-dev-multikilt_1000_k20_dep3"
FOLDER_RE = re.compile(r"^(hotpotqa|nq|triviaqa)-dev-multikilt_1000_k(20|50|105|220|440|1000)_dep(3|6)$")


def load_scores(base_dir: str = BASE_DIR, metric_name="doc_ids_f1"):
    """Return nested dict: scores[benchmark][model][context_label] = list of scores."""
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for entry in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, entry)
        if not os.path.isdir(folder_path):
            continue

        m = FOLDER_RE.match(entry)
        if not m:
            # Skip unrelated directories
            continue

        benchmark, num_retrieved, depth = m.groups()
        context_label = CONTEXT_MAP.get(num_retrieved)
        if context_label is None:
            continue

        # Each subdirectory here should be a model name
        for model in os.listdir(folder_path):
            model_path = os.path.join(folder_path, model)
            if not os.path.isdir(model_path):
                continue

            json_path = os.path.join(model_path, "zero_shot_doc_ids.json")
            if not os.path.isfile(json_path):
                json_path = os.path.join(model_path, "zero_shot_quote.json")
                if not os.path.isfile(json_path):
                    json_path = os.path.join(model_path, "zero_shot_doc_ids_and_content.json")
                    if not os.path.isfile(json_path):
                        continue
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                score = data.get(metric_name, None)
                if score is None:
                    continue
                # Ensure numeric
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    continue
                scores[benchmark][model][context_label].append(score)
            except Exception as e:
                print(f"Warning: failed to read {json_path}: {e}")

    return scores


def plot_scores(scores, metric_name="doc_ids_f1"):
    """Plot one chart per benchmark, one line per model."""
    for benchmark, models_dict in scores.items():
        plt.figure(figsize=(9, 6))

        # Stable legend order
        for model in sorted(models_dict.keys()):
            ctx_to_vals = models_dict[model]
            y_vals = []
            for ctx in CONTEXT_ORDER:
                vals = ctx_to_vals.get(ctx, [])
                if vals:
                    # Average any duplicates (e.g., 440 & 500 => both into 64k)
                    avg = sum(vals) / len(vals)
                    y_vals.append(avg)
                else:
                    y_vals.append(None)  # keep spacing, show gaps if missing

            plt.plot(CONTEXT_ORDER, y_vals, marker="o", label=model)

        if "f1" in metric_name:
            plt.title(f"HELMET - Doc IDs F1 on {benchmark}")
            plt.xlabel("Context Length")
            plt.ylabel("F1 score")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Model", loc="best")
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}/HELMET_doc_f1_{benchmark}.png", format="png")
        elif "precision" in metric_name:
            plt.title(f"HELMET - Doc IDs Precision on {benchmark}")
            plt.xlabel("Context Length")
            plt.ylabel("Precision")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Model", loc="best")
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}/HELMET_doc_precision_{benchmark}.png", format="png")
        elif "recall" in metric_name:
            plt.title(f"HELMET - Doc IDs Recall on {benchmark}")
            plt.xlabel("Context Length")
            plt.ylabel("Recall")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(title="Model", loc="best")
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}/HELMET_doc_recall_{benchmark}.png", format="png")
        else:
            raise ValueError(f"Not a valid metric: {metric_name}")

if __name__ == "__main__":
    f1_scores = load_scores(metric_name="doc_ids_f1")
    precision_scores = load_scores(metric_name="doc_ids_precision")
    recall_scores = load_scores(metric_name="doc_ids_recall")
    plot_scores(f1_scores, metric_name="doc_ids_f1")
    plot_scores(precision_scores, metric_name="doc_ids_precision")
    plot_scores(recall_scores, metric_name="doc_ids_recall")