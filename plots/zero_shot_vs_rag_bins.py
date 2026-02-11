# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json, re
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# COMMAND LINE ARGUMENT PARSING
def parse_args():
    parser = argparse.ArgumentParser(description='Plot benchmark results with RAG analysis')
    parser.add_argument('--benchmark-name', '-b', default='InfiniteBench', 
                       help='Benchmark name (default: InfiniteBench)')
    parser.add_argument('--benchmark-split', '-s', default='longbook_choice_eng',
                       help='Benchmark split (default: longbook_choice_eng)')
    parser.add_argument('--exclude-models', '-e', nargs='*', default=[],
                       help='List of model names to exclude from plots')
    parser.add_argument('--metric', default='accuracy',
                       help='Metric to track (default: accuracy)')
    return parser.parse_args()

args = parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
benchmark_name  = args.benchmark_name
benchmark_split = args.benchmark_split
excluded_models = set(args.exclude_models)
metric_name     = args.metric
root            = Path("data/evaluation") / benchmark_name / benchmark_split
rag_token_limits = (4, 8, 16, 32, 64, 128)          # → rag_4k … rag_128k
# ──────────────────────────────────────────────────────────────────────────────


# Regular expressions reused for every model
rx_range   = re.compile(r"maximum_input_length=\[(\d+),(\d+)\]")
rx_rag_tok = re.compile(r"rag_context_max_tokens=(\d+)")

def build_series_dict() -> dict[str, dict[int, float]]:
    """Return an empty `series_bins` dict for a new model."""
    return {
        "fc" : {},
        "rag": {},
        **{f"rag_{k}k": {} for k in rag_token_limits},
    }

def add_file_to_series(jf: Path, series_bins: dict[str, dict[int, float]]):
    """Parse one JSON filename + body and put its metric value into `series_bins`."""
    m_range = rx_range.search(jf.stem)
    if not m_range:
        return                                      # skip files with no range info

    _, max_len = map(int, m_range.groups())         # use *upper* bound as the bin key
    bin_key = max_len

    stem = jf.stem
    if "rag_context" in stem:
        m_tok = rx_rag_tok.search(stem)
        if m_tok:
            tok_k = int(m_tok.group(1)) // 1000
            curve = f"rag_{tok_k}k"
        else:
            curve = "rag"
    elif "rag" in stem:
        curve = "rag"
    else:
        curve = "fc"

    try:
        with jf.open() as fp:
            metric_value = json.load(fp)[metric_name]
        series_bins[curve][bin_key] = metric_value
    except (json.JSONDecodeError, KeyError):
        print(f"Skipping {jf} - malformed JSON or no '{metric_name}' field.")

def plot_model(model_name: str, series_bins: dict[str, dict[int, float]]):
    """Create and save one plot for the given model."""
    all_bins = sorted({b for d in series_bins.values() for b in d})
    if not all_bins:            # no data at all → nothing to plot
        print(f"{model_name}: no JSON files that match the pattern - skipping")
        return

    x_labels = [round(b/1000) for b in all_bins]
    x_pos    = range(len(all_bins))

    def build(curve):           # helper that emits a list aligned with all_bins
        d = series_bins.get(curve, {})
        return [d.get(b, np.nan) for b in all_bins]

    fc_values       = build("fc")
    rag_values      = build("rag")
    rag_128k_values = build("rag_128k")
    rag_64k_values  = build("rag_64k")
    rag_32k_values  = build("rag_32k")
    rag_16k_values  = build("rag_16k")
    rag_8k_values   = build("rag_8k")
    rag_4k_values   = build("rag_4k")

    fig, ax = plt.subplots(figsize=(8, 5))

    curves = [
        ("FC",          fc_values,      'o'),
        ("RAG",         rag_values,     's'),
        ("RAG @ 128k", rag_128k_values, 's'),
        ("RAG @ 64k",  rag_64k_values,  's'),
        ("RAG @ 32k",  rag_32k_values,  's'),
        ("RAG @ 16k",  rag_16k_values,  's'),
        ("RAG @ 8k",   rag_8k_values,   's'),
        ("RAG @ 4k",   rag_4k_values,   's'),
    ]

    for label, vals, marker in curves:
        if np.isfinite(vals).any():      # show only curves that have ≥1 real point
            ax.plot(x_pos, vals, marker=marker, label=label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Tokens (thousands)')

    # autoscale Y using finite points only
    finite = np.array(
        sum((v for _, v, _ in curves), []),
        dtype=float
    )
    finite = finite[np.isfinite(finite)]
    padding = max(1, 0.05 * (finite.max() - finite.min()))
    ax.set_ylim(finite.min() - padding, finite.max() + padding)

    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f'{benchmark_name}-choice - {model_name}')
    ax.grid(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0), prop={'size': 6}, frameon=True)
    fig.tight_layout()

    outfile = f"{benchmark_name}_{benchmark_split}_{model_name}_rag_bins.png"
    fig.savefig(outfile, dpi=400)
    plt.close(fig)
    print(f"Wrote {outfile}")


# ─────────────────────── main loop: one plot per model ────────────────────────
for model_dir in sorted(root.iterdir()):
    if not model_dir.is_dir():
        continue

    model_name = model_dir.name
    
    # Skip excluded models
    if model_name in excluded_models:
        print(f"Skipping excluded model: {model_name}")
        continue
    
    series_bins = build_series_dict()

    for jf in model_dir.glob("*.json"):
        add_file_to_series(jf, series_bins)

    plot_model(model_name, series_bins)