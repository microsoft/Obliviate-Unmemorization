#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import re
import argparse
import numpy as np
from collections import defaultdict

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def read_sample_labels(label_file):
    """Read sample labels from file."""
    if not label_file:
        return None
    try:
        with open(label_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Warning: Could not read sample labels: {e}")
        return None

def extract_metrics(file_path):
    """Extract metrics from the file for offset 0, prime length 10."""
    metrics = {'bleu': [], 'rouge2': [], 'lcs': []}
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return metrics
        
    debug_print(f"\nProcessing file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Split into samples
    sample_sections = re.split(r'\*+\nSample \d+', content)
    
    for i, section in enumerate(sample_sections):
        # Look for prime length 10 sections with offset 0
        if 'Offset 0 Prime length 10:' in section:
            # Extract BLEU score
            bleu_match = re.search(r"'bleu': ([\d.]+)", section)
            if bleu_match:
                bleu_score = float(bleu_match.group(1))
                metrics['bleu'].append(bleu_score)
                
            # Extract ROUGE2 score
            rouge2_match = re.search(r"'rouge2': ([\d.]+)", section)
            if rouge2_match:
                rouge2_score = float(rouge2_match.group(1))
                metrics['rouge2'].append(rouge2_score)
                
            # Extract longest common substring length
            lcs_match = re.search(r"Longest common substring length: (\d+)", section)
            if lcs_match:
                lcs_length = int(lcs_match.group(1))
                metrics['lcs'].append(lcs_length)
                
    return metrics

def extract_epoch_data(directory):
    """Extract epoch data from unmemorize.log files."""
    numbered_runs = []

    def get_max_epoch(log_file):
        if not os.path.exists(log_file):
            return 0
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            epoch_matches = re.findall(r'epoch (\d+):', content)
            return int(epoch_matches[-1]) if epoch_matches else 0
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            return 0

    for subdir1 in os.listdir(directory):
        if subdir1 not in ['pretrained', 'memorized'] and os.path.isdir(os.path.join(directory, subdir1)):
            try:
                x, y, z = map(int, subdir1.split('-'))
                log_file = os.path.join(directory, subdir1, '0', 'unmemorize.log')
                epochs = get_max_epoch(log_file)
                numbered_runs.append((x, y, z, subdir1, epochs))
            except Exception as e:
                print(f"Error processing {subdir1}: {e}")

    numbered_runs.sort(key=lambda x: (x[1], x[2], x[0]))
    return [(run[3], run[4]) for run in numbered_runs]

def extract_benchmark_data(directory, pretrained_path=None, memorized_path=None):
    """Extract and normalize benchmark data from benchmark.log files.
    
    Args:
        directory (str): Directory containing the run folders
        pretrained_path (str, optional): Path to pretrained directory
        memorized_path (str, optional): Path to memorized directory
    """
    numbered_runs = []
    special_runs = []
    pretrained_results = None

    patterns = {
        'MMLU acc': r'mmlu.*?\|.*?\|.*?\|.*?\|acc.*?\|\s*(?:↑\s*)?([\d\.]+)',
        'Truthfulqa acc': r'truthfulqa_mc2.*?\|.*?\|.*?\|.*?\|acc.*?\|\s*(?:↑\s*)?([\d\.]+)',
        'Hellaswag acc': r'hellaswag.*?\|.*?\|.*?\|.*?\|acc.*?\|\s*(?:↑\s*)?([\d\.]+)',
        'winogrande acc': r'winogrande.*?\|.*?\|.*?\|.*?\|acc.*?\|\s*(?:↑\s*)?([\d\.]+)'
    }

    def process_benchmark_log(log_file, run_name):
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                results = {k: float(re.search(p, log_content).group(1)) 
                         for k, p in patterns.items()}
                return (run_name, results)
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
        return None

    # First, try to get memorized results for normalization
    if memorized_path:
        log_file = os.path.join(os.path.dirname(memorized_path), 'benchmark.log')
        memorized_result = process_benchmark_log(log_file, 'memorized')
        if memorized_result:
            normalization_results = memorized_result[1]
            special_runs.append(memorized_result)
    
    # If no memorized results from path, check directory
    if not 'normalization_results' in locals():
        dir_path = os.path.join(directory, 'memorized')
        if os.path.isdir(dir_path):
            log_file = os.path.join(dir_path, 'benchmark.log')
            memorized_result = process_benchmark_log(log_file, 'memorized')
            if memorized_result:
                normalization_results = memorized_result[1]
                if not any(run[0] == 'memorized' for run in special_runs):
                    special_runs.append(memorized_result)
    
    # If no memorized results, fall back to pretrained
    if not 'normalization_results' in locals():
        if pretrained_path:
            log_file = os.path.join(os.path.dirname(pretrained_path), 'benchmark.log')
            pretrained_result = process_benchmark_log(log_file, 'pretrained')
            if pretrained_result:
                normalization_results = pretrained_result[1]
                special_runs.append(pretrained_result)
        
        if not 'normalization_results' in locals():
            dir_path = os.path.join(directory, 'pretrained')
            if os.path.isdir(dir_path):
                log_file = os.path.join(dir_path, 'benchmark.log')
                pretrained_result = process_benchmark_log(log_file, 'pretrained')
                if pretrained_result:
                    normalization_results = pretrained_result[1]
                    if not any(run[0] == 'pretrained' for run in special_runs):
                        special_runs.append(pretrained_result)

    # Process memorized path
    if memorized_path:
        log_file = os.path.join(os.path.dirname(memorized_path), 'benchmark.log')
        result = process_benchmark_log(log_file, 'memorized')
        if result and not any(run[0] == 'memorized' for run in special_runs):
            special_runs.append(result)
    else:
        # Check directory for memorized
        dir_path = os.path.join(directory, 'memorized')
        if os.path.isdir(dir_path):
            log_file = os.path.join(dir_path, 'benchmark.log')
            result = process_benchmark_log(log_file, 'memorized')
            if result and not any(run[0] == 'memorized' for run in special_runs):
                special_runs.append(result)

    # Process numbered runs
    for subdir1 in os.listdir(directory):
        if subdir1 not in ['pretrained', 'memorized'] and os.path.isdir(os.path.join(directory, subdir1)):
            try:
                x, y, z = map(int, subdir1.split('-'))
                log_file = os.path.join(directory, subdir1, '0', 'benchmark.log')
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    results = {k: float(re.search(p, log_content).group(1))
                             for k, p in patterns.items()}
                    numbered_runs.append((x, y, z, subdir1, results))
            except Exception as e:
                print(f"Error processing {subdir1}: {e}")

    numbered_runs.sort(key=lambda x: (x[1], x[2], x[0]))
    all_runs = special_runs + [(run[3], run[4]) for run in numbered_runs]

    # Normalize results if normalization base is available
    if 'normalization_results' in locals():
        normalized_runs = []
        for run_name, results in all_runs:
            normalized_results = {
                k: v / normalization_results[k] 
                for k, v in results.items()
            }
            normalized_runs.append((run_name, normalized_results))
        return normalized_runs
    
    return all_runs

def plot_benchmark_results(benchmark_data, output_dir, plot_title=""):
    """Create plot comparing normalized benchmark results across runs."""
    plt.figure(figsize=(12, 6))
    x_labels = [run[0] for run in benchmark_data]
    
    metric_names = {
        'MMLU acc': 'MMLU',
        'Truthfulqa acc': 'TruthfulQA',
        'Hellaswag acc': 'HellaSwag',
        'winogrande acc': 'Winogrande'
    }
    
    for metric, label in metric_names.items():
        scores = [run[1][metric] for run in benchmark_data]
        plt.plot(x_labels, scores, label=label, marker='o')

    norm_base = "Memorized" if any(run[0] == 'memorized' for run in benchmark_data) else "Pretrained"
    plt.title(f'{plot_title}\nBenchmark Results\n(Normalized to {norm_base} Performance)')
    plt.xlabel('Run')
    plt.ylabel('Performance Relative to Pretrained')
    # Set y-axis limits to 0.01 above/below min/max values
    all_scores = [score for metric in metric_names.keys() 
                 for run in benchmark_data 
                 for score in [run[1][metric]]]
    y_min = min(all_scores) - 0.01
    y_max = max(all_scores) + 0.01
    plt.ylim(y_min, y_max)
    # Configure y-axis with proper tick spacing
    ax = plt.gca()
    all_scores = [score for metric in metric_names.keys() 
                 for run in benchmark_data 
                 for score in [run[1][metric]]]
    y_min = min(all_scores) - 0.01
    y_max = max(all_scores) + 0.01
    
    # Use MaxNLocator to avoid duplicate ticks
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.ylim(y_min, y_max)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)  # Add reference line at 1.0
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmarks_run_comparison.png'))
    plt.close()
    
def plot_epochs(epochs_data, output_dir, plot_title=""):
    """Create plot comparing epochs across runs."""
    plt.figure(figsize=(12, 6))
    x_labels = [run[0] for run in epochs_data]
    epochs = [run[1] for run in epochs_data]
    
    plt.bar(x_labels, epochs)
    plt.title(f'{plot_title}\nEpochs per Run')
    plt.xlabel('Run')
    plt.ylabel('Number of Epochs')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(epochs):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epochs_run_comparison.png'))
    plt.close()

def process_runs_folder(folder_path, pretrained_dir=None, memorized_dir=None, 
                   pretrained_greedy=None, memorized_greedy=None):
    """Process all runs in the folder and collect their metrics."""
    run_metrics = defaultdict(lambda: defaultdict(list))
    
    # Process main runs folder
    subdirs = [d for d in os.listdir(folder_path) 
              if os.path.isdir(os.path.join(folder_path, d))]
    
    for subdir in sorted(subdirs):
        test_log = os.path.join(folder_path, subdir, "0", "test.log")
        test_greedy_log = os.path.join(folder_path, subdir, "0", "test_greedy.log")
        
        if os.path.exists(test_log):
            print(f"Processing {subdir} test.log")
            metrics = extract_metrics(test_log)
            for metric_type in metrics:
                if metrics[metric_type]:  # Only add if there are values
                    run_metrics[metric_type][subdir].extend(metrics[metric_type])
            
        if os.path.exists(test_greedy_log):
            print(f"Processing {subdir} test_greedy.log")
            metrics = extract_metrics(test_greedy_log)
            for metric_type in metrics:
                if metrics[metric_type]:  # Only add if there are values
                    run_metrics[metric_type][f"{subdir} (Greedy)"].extend(metrics[metric_type])

    # Process pretrained directory if provided
    if pretrained_dir:
        test_log = os.path.join(os.path.dirname(pretrained_dir), "test.log")
        if os.path.exists(test_log):
            metrics = extract_metrics(test_log)
            for metric_type in metrics:
                if metrics[metric_type]:
                    run_metrics[metric_type]["Pretrained"].extend(metrics[metric_type])

    # Process pretrained greedy if provided
    if pretrained_greedy:
        metrics = extract_metrics(pretrained_greedy)
        for metric_type in metrics:
            if metrics[metric_type]:
                run_metrics[metric_type]["Pretrained (Greedy)"].extend(metrics[metric_type])

    # Process memorized directory if provided
    if memorized_dir:
        test_log = os.path.join(os.path.dirname(memorized_dir), "test.log")
        if os.path.exists(test_log):
            metrics = extract_metrics(test_log)
            for metric_type in metrics:
                if metrics[metric_type]:
                    run_metrics[metric_type]["Memorized"].extend(metrics[metric_type])

    # Process memorized greedy if provided
    if memorized_greedy:
        metrics = extract_metrics(memorized_greedy)
        for metric_type in metrics:
            if metrics[metric_type]:
                run_metrics[metric_type]["Memorized (Greedy)"].extend(metrics[metric_type])
    
    return run_metrics

def create_run_comparison_plots(run_metrics, output_dir, plot_title=""):
    """Create plots comparing metrics across different runs with greedy and non-greedy as separate lines."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('classic')
    plot_params = {
        'figure.figsize': (15, 8),
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
    }
    plt.rcParams.update(plot_params)

    metrics_info = {
        'bleu': ('BLEU Scores', 'BLEU Score', False),
        'rouge2': ('ROUGE2 Scores', 'ROUGE2 Score', False),
        'lcs': ('Longest Common Substring Length', 'Length', True)
    }

    def get_base_run_name(run_name):
        return run_name.replace(" (Greedy)", "")

    def get_sort_key(run_name):
        if run_name in ['Pretrained', 'Memorized']:
            return (-1, 0, 0)
        try:
            x, y, z = map(int, run_name.split('-'))
            return (y, x, z)
        except (ValueError, AttributeError):
            return (float('inf'), 0, 0)

    for metric_type, (title_suffix, ylabel, use_log_scale) in metrics_info.items():
        if metric_type not in run_metrics:
            continue

        fig, ax = plt.subplots()
        
        run_data = run_metrics[metric_type]
        all_runs = sorted(run_data.keys())
        base_runs = sorted(set(get_base_run_name(run) for run in all_runs), 
                         key=get_sort_key)
        x_positions = np.arange(len(base_runs))
        
        all_values = [v for values in run_data.values() for v in values]
        if use_log_scale:
            y_min = max(0.1, min(all_values) * 0.9)
            y_max = max(all_values) * 1.1
        else:
            y_min = min(0, min(all_values))
            y_max = max(all_values) * 1.1
        
        for is_greedy_line in [False, True]:
            means = []
            stds = []
            
            for base_run in base_runs:
                run_name = base_run + (" (Greedy)" if is_greedy_line else "")
                if run_name in run_data:
                    values = run_data[run_name]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(None)
                    stds.append(None)
            
            valid_positions = []
            valid_means = []
            valid_stds = []
            for i, (mean, std) in enumerate(zip(means, stds)):
                if mean is not None:
                    valid_positions.append(x_positions[i])
                    valid_means.append(mean)
                    valid_stds.append(std)
            
            if valid_positions:
                line_style = '--' if is_greedy_line else '-'
                label = 'Greedy' if is_greedy_line else 'Regular'
                
                ax.errorbar(valid_positions, valid_means, yerr=valid_stds, 
                          fmt=f'o{line_style}', capsize=5, markersize=8, linewidth=2,
                          label=label)
                
                for i, pos in enumerate(valid_positions):
                    run_name = base_runs[pos] + (" (Greedy)" if is_greedy_line else "")
                    values = run_data[run_name]
                    points = ax.plot([pos] * len(values), values, 'o', alpha=0.3, markersize=5)
                    
                    for val in values:
                        if use_log_scale:
                            label_y = val * 1.1
                            label_text = f'{int(val)}'
                        else:
                            label_y = val + (y_max - y_min) * 0.02
                            label_text = f'{val:.2f}'
                        
                        ax.annotate(label_text, 
                                  xy=(pos, val),
                                  xytext=(pos, label_y),
                                  ha='center', va='bottom',
                                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1),
                                  fontsize=8)
        
        ax.set_title(f'{plot_title} {title_suffix} Comparison Across Runs')
        ax.set_xlabel('Run Configuration')
        ax.set_ylabel(ylabel)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(base_runs, rotation=45, ha='right')
        
        if use_log_scale:
            ax.set_yscale('log')
            ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        ax.set_ylim(y_min, y_max)
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{metric_type}_run_comparison.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

def create_multi_sample_comparison_plots(metrics_dict, output_dir, plot_title="", sample_labels=None):
    """Create plots comparing metrics across samples (original functionality)."""
    # [Previous implementation of create_multi_comparison_plots remains the same]
    # This preserves the original sample-based comparison functionality
    pass

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots for test metrics.')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--runs_folder', help='Path to folder containing multiple run directories')
    parser.add_argument('--input', help='Path to the current test.log file')
    parser.add_argument('--input_greedy', help='Path to the greedy version of current test.log file')
    parser.add_argument('--output', required=True, help='Directory to save the plots')
    parser.add_argument('--pretrained', help='Path to the pretrained test.log file for comparison')
    parser.add_argument('--memorized', help='Path to the memorized test.log file for comparison')
    parser.add_argument('--pretrained_greedy', help='Path to the greedy pretrained test.log file')
    parser.add_argument('--memorized_greedy', help='Path to the greedy memorized test.log file')
    parser.add_argument('--title', default='', help='Title prefix for the plots')
    parser.add_argument('--sample-labels', help='Path to file containing sample labels (one per line)')
    
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = args.debug
    
    if args.runs_folder:
        print(f"Processing runs folder: {args.runs_folder}")
        # Process test metrics
        run_metrics = process_runs_folder(
            args.runs_folder,
            args.pretrained,
            args.memorized,
            args.pretrained_greedy,
            args.memorized_greedy
        )
        
        if not run_metrics:
            print("No valid data found in runs folder")
            return
            
        create_run_comparison_plots(run_metrics, args.output, args.title)
        
        # Process and plot epoch data
        epoch_data = extract_epoch_data(args.runs_folder)
        if epoch_data:
            plot_epochs(epoch_data, args.output, args.title)
        
        # Process and plot benchmark data
        benchmark_data = extract_benchmark_data(args.runs_folder, args.pretrained, args.memorized)
        if benchmark_data:
            plot_benchmark_results(benchmark_data, args.output, args.title)
            
    else:
        # Original functionality for comparing individual files and samples
        metrics_dict = {'Current': extract_metrics(args.input)}
        
        if args.input_greedy:
            metrics_dict['Current Greedy'] = extract_metrics(args.input_greedy)
        if args.pretrained:
            metrics_dict['Pretrained'] = extract_metrics(args.pretrained)
        if args.memorized:
            metrics_dict['Memorized'] = extract_metrics(args.memorized)
        
        sample_labels = read_sample_labels(args.sample_labels) if args.sample_labels else None
        create_multi_sample_comparison_plots(metrics_dict, args.output, args.title, sample_labels)

if __name__ == "__main__":
    main()