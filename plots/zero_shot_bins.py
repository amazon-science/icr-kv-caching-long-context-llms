# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot zero-shot performance for Qwen2.5-7B-Instruct-1M models across input length bins',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py --benchmark InfiniteBench --split longbook_choice_eng
  python script.py --benchmark HELMET --split kilt --modality zero_shot_doc_ids
  python script.py --benchmark InfiniteBench --split longbook_qa_eng --exclude Qwen2.5-7B-Instruct-1M-GRPO Qwen2.5-7B-Instruct-1M-SFT
        """)
    
    parser.add_argument(
        '--benchmark', '-b',
        type=str,
        default='InfiniteBench',
        help='Benchmark name (default: InfiniteBench)'
    )
    
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='longbook_choice_eng',
        help='Benchmark split/subset (default: longbook_choice_eng)'
    )
    
    parser.add_argument(
        '--modality', '-m',
        type=str,
        default='zero_shot',
        help='Zero-shot modality: zero_shot, zero_shot_RetroInfer, zero_shot_doc_ids, etc. (default: zero_shot)'
    )
    
    parser.add_argument(
        '--exclude', '-e',
        nargs='*',
        default=[],
        help='Model names to exclude (space-separated list)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/evaluation',
        help='Root data directory (default: data/evaluation)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='accuracy',
        help='Metric to extract from JSON files (default: accuracy)'
    )
    
    return parser.parse_args()

# Regular expression to parse maximum_input_length ranges
rx_range = re.compile(r"maximum_input_length=\[(\d+),(\d+)\]")

def collect_model_data(root, modality, exclude_models, metric):
    """Collect performance data for Qwen2.5-7B-Instruct-1M models only."""
    model_data = {}
    
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Filter to only Qwen2.5-7B-Instruct-1M models
        if not model_name.startswith("Qwen2.5-7B-Instruct-1M"):
            continue
            
        # Skip excluded models
        if model_name in exclude_models:
            print(f"Excluding model: {model_name}")
            continue
            
        print(f"Processing model: {model_name}")
        model_data[model_name] = {}
        
        # Look for specified modality files with maximum_input_length ranges
        pattern = f"{modality}_maximum_input_length=*.json"
        modality_files = list(model_dir.glob(pattern))
        print(f"  Found {len(modality_files)} binned files for modality '{modality}'")
        
        for json_file in modality_files:
            print(f"  Processing: {json_file.name}")
            m_range = rx_range.search(json_file.stem)
            if not m_range:
                print(f"    No range match in: {json_file.stem}")
                continue
                
            _, max_len = map(int, m_range.groups())  # Use upper bound as bin key
            bin_key = max_len
            
            try:
                with json_file.open() as fp:
                    data = json.load(fp)
                    print(f"    JSON keys: {list(data.keys())}")
                    if isinstance(data, dict):
                        print(f"    JSON sample: {str(data)[:200]}...")
                    
                    if metric in data:
                        model_data[model_name][bin_key] = data[metric]
                        print(f"    Added bin {bin_key}: {metric} = {data[metric]}")
                    else:
                        print(f"    No '{metric}' key found in {json_file}")
                        # Try other possible keys
                        for key in data.keys():
                            if 'acc' in key.lower() or 'score' in key.lower() or 'f1' in key.lower():
                                print(f"    Possible metric key found: {key} = {data[key]}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    Error reading {json_file}: {e}")
            except Exception as e:
                print(f"    Unexpected error with {json_file}: {e}")
                
        # Also check for basic modality file (unbinned data)
        basic_file = model_dir / f"{modality}.json"
        if basic_file.exists():
            try:
                with basic_file.open() as fp:
                    data = json.load(fp)
                    if metric in data:
                        print(f"  Found overall {modality}.json: {metric} = {data[metric]}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Error reading {modality}.json: {e}")
    
    return model_data

def plot_multi_model_performance(model_data, benchmark_name, benchmark_split, modality, metric):
    """Create a plot showing performance for all models across input length bins."""
    
    # Collect all unique bins across all models
    all_bins = set()
    for model_bins in model_data.values():
        all_bins.update(model_bins.keys())
    
    # Remove infinity if present and sort
    finite_bins = sorted([b for b in all_bins if np.isfinite(b)])
    
    if not finite_bins:
        print("No binned data found - skipping plot")
        return
        
    # Convert bins to thousands for x-axis labels
    x_labels = [f"{int(b/1000)}k" for b in finite_bins]
    x_pos = range(len(finite_bins))
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors/markers for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot each model
    plotted_models = 0
    for i, (model_name, model_bins) in enumerate(model_data.items()):
        if not model_bins:
            print(f"No data for {model_name}")
            continue
            
        # Get metric values for each bin
        y_values = [model_bins.get(bin_key, np.nan) for bin_key in finite_bins]
        
        # Only plot if there's at least one valid data point
        valid_points = np.isfinite(y_values).sum()
        if valid_points == 0:
            print(f"No valid data points for {model_name}")
            continue
            
        print(f"Plotting {model_name} with {valid_points} data points")
        
        # Plot the line
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(x_pos, y_values, 
               marker=marker, 
               color=color, 
               label=model_name, 
               linewidth=2, 
               markersize=8,
               alpha=0.8)
        plotted_models += 1
    
    # Customize the plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Maximum Input Length (tokens)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{benchmark_name} {benchmark_split} - {modality.replace("_", " ").title()} {metric.replace("_", " ").title()} by Input Length', 
                fontsize=14, fontweight='bold')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    
    if plotted_models > 0:
        # Place legend inside the plot area
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
    else:
        print("No models were plotted!")
        return
    
    # Tight layout to prevent cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = f"{benchmark_name}_{benchmark_split}_{modality}_{metric}.png"
    fig.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    plt.show()

def print_summary(model_data, benchmark_name, benchmark_split, metric):
    """Print a summary of the collected data."""
    print(f"\nData Summary for {benchmark_name} {benchmark_split}:")
    print("=" * 60)
    
    for model_name, model_bins in model_data.items():
        if model_bins:
            finite_bins = [b for b in model_bins.keys() if np.isfinite(b)]
            bin_count = len(finite_bins)
            metric_values = [model_bins[b] for b in finite_bins if np.isfinite(model_bins[b])]
            if metric_values:
                avg_metric = np.mean(metric_values)
                print(f"{model_name:40s}: {bin_count:2d} bins, avg {metric}: {avg_metric:.3f}")
            else:
                print(f"{model_name:40s}: {bin_count:2d} bins, no valid {metric} values")
        else:
            print(f"{model_name:40s}: No data found")

# ─────────────────────── Main execution ────────────────────────
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up paths
    root = Path(args.data_dir) / args.benchmark / args.split
    
    print(f"Processing benchmark: {args.benchmark}/{args.split}")
    print(f"Modality: {args.modality}")
    print(f"Metric: {args.metric}")
    print(f"Looking in directory: {root}")
    if args.exclude:
        print(f"Excluding models: {', '.join(args.exclude)}")
    
    if not root.exists():
        print(f"Error: Directory {root} does not exist!")
        exit(1)
    
    # Collect data from all models
    model_data = collect_model_data(root, args.modality, args.exclude, args.metric)
    
    if not model_data:
        print("No model data found!")
        exit(1)
    
    # Print summary
    print_summary(model_data, args.benchmark, args.split, args.metric)
    
    # Create the plot
    plot_multi_model_performance(model_data, args.benchmark, args.split, args.modality, args.metric)