"""
Compare and visualize model extraction attack results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse


def load_results(results_dir):
    """Load all result files"""
    results_dir = Path(results_dir)
    all_results = []
    
    for results_file in sorted(results_dir.glob("results_*samples.json")):
        with open(results_file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    
    return all_results


def create_comparison_table(all_results):
    """Create comparison table"""
    
    rows = []
    for result in all_results:
        train_samples = result['config']['train_samples']
        dataset = result['config']['dataset']
        
        # Add victim
        victim = result['victim']
        rows.append({
            'Dataset': dataset,
            'Training Samples': '-',
            'Model': victim['model'] + ' (victim)',
            'Perplexity': f"{victim['perplexity']:.2f}",
            'Accuracy (%)': f"{victim['accuracy']:.2f}",
            'Fidelity@top-1 (%)': f"{victim['fidelity@top-1']:.2f}"
        })
        
        # Add baseline
        baseline = result['baseline']
        rows.append({
            'Dataset': dataset,
            'Training Samples': '-',
            'Model': baseline['model'],
            'Perplexity': f"{baseline['perplexity']:.2f}",
            'Accuracy (%)': f"{baseline['accuracy']:.2f}",
            'Fidelity@top-1 (%)': f"{baseline['fidelity@top-1']:.2f}"
        })
        
        # Add finetuned
        finetuned = result['finetuned']
        rows.append({
            'Dataset': dataset,
            'Training Samples': str(train_samples),
            'Model': finetuned['model'],
            'Perplexity': f"{finetuned['perplexity']:.2f}",
            'Accuracy (%)': f"{finetuned['accuracy']:.2f}",
            'Fidelity@top-1 (%)': f"{finetuned['fidelity@top-1']:.2f}"
        })
        
        rows.append({
            'Dataset': '',
            'Training Samples': '',
            'Model': '',
            'Perplexity': '',
            'Accuracy (%)': '',
            'Fidelity@top-1 (%)': ''
        })
    
    df = pd.DataFrame(rows)
    return df


def plot_metrics_comparison(all_results, output_dir):
    """Plot comparison of metrics across different training sizes"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    train_sizes = []
    baseline_ppls = []
    finetuned_ppls = []
    baseline_accs = []
    finetuned_accs = []
    baseline_fids = []
    finetuned_fids = []
    
    for result in sorted(all_results, key=lambda x: x['config']['train_samples']):
        train_size = result['config']['train_samples']
        train_sizes.append(train_size)
        
        baseline_ppls.append(result['baseline']['perplexity'])
        finetuned_ppls.append(result['finetuned']['perplexity'])
        
        baseline_accs.append(result['baseline']['accuracy'])
        finetuned_accs.append(result['finetuned']['accuracy'])
        
        baseline_fids.append(result['baseline']['fidelity@top-1'])
        finetuned_fids.append(result['finetuned']['fidelity@top-1'])
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 5)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Perplexity
    axes[0].plot(train_sizes, baseline_ppls, 'o-', label='Baseline DistilGPT-2', linewidth=2, markersize=8)
    axes[0].plot(train_sizes, finetuned_ppls, 's-', label='Fine-tuned DistilGPT-2', linewidth=2, markersize=8)
    if len(all_results) > 0:
        victim_ppl = all_results[0]['victim']['perplexity']
        axes[0].axhline(y=victim_ppl, color='r', linestyle='--', label='Victim GPT-2', linewidth=2)
    axes[0].set_xlabel('Training Samples', fontsize=12)
    axes[0].set_ylabel('Perplexity', fontsize=12)
    axes[0].set_title('Perplexity vs Training Size', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(train_sizes, baseline_accs, 'o-', label='Baseline DistilGPT-2', linewidth=2, markersize=8)
    axes[1].plot(train_sizes, finetuned_accs, 's-', label='Fine-tuned DistilGPT-2', linewidth=2, markersize=8)
    if len(all_results) > 0:
        victim_acc = all_results[0]['victim']['accuracy']
        axes[1].axhline(y=victim_acc, color='r', linestyle='--', label='Victim GPT-2', linewidth=2)
    axes[1].set_xlabel('Training Samples', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy vs Training Size', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Fidelity
    axes[2].plot(train_sizes, baseline_fids, 'o-', label='Baseline DistilGPT-2', linewidth=2, markersize=8)
    axes[2].plot(train_sizes, finetuned_fids, 's-', label='Fine-tuned DistilGPT-2', linewidth=2, markersize=8)
    axes[2].axhline(y=100, color='r', linestyle='--', label='Victim GPT-2', linewidth=2)
    axes[2].set_xlabel('Training Samples', fontsize=12)
    axes[2].set_ylabel('Fidelity@top-1 (%)', fontsize=12)
    axes[2].set_title('Fidelity@top-1 vs Training Size', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'extraction_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    
    plt.show()


def plot_training_curves(all_results, output_dir):
    """Plot training loss curves"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, len(all_results), figsize=(8*len(all_results), 5))
    
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, result in enumerate(sorted(all_results, key=lambda x: x['config']['train_samples'])):
        train_size = result['config']['train_samples']
        history = result['training_history']
        
        epochs = [h['epoch'] for h in history]
        total_losses = [h['loss'] for h in history]
        kl_losses = [h['kl_loss'] for h in history]
        ce_losses = [h['ce_loss'] for h in history]
        
        axes[idx].plot(epochs, total_losses, 'o-', label='Total Loss', linewidth=2, markersize=8)
        axes[idx].plot(epochs, kl_losses, 's-', label='KL Divergence Loss', linewidth=2, markersize=8)
        axes[idx].plot(epochs, ce_losses, '^-', label='Cross Entropy Loss', linewidth=2, markersize=8)
        
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel('Loss', fontsize=12)
        axes[idx].set_title(f'Training Curves ({train_size} samples)', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'training_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    
    plt.show()


def plot_metrics_heatmap(all_results, output_dir):
    """Create heatmap of metrics"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    data = []
    labels = []
    
    for result in sorted(all_results, key=lambda x: x['config']['train_samples']):
        train_size = result['config']['train_samples']
        
        baseline = result['baseline']
        finetuned = result['finetuned']
        
        # Normalize metrics for better visualization
        # Lower perplexity is better, so we use 1/perplexity
        baseline_norm = [
            1.0 / baseline['perplexity'] * 100,  # Inverse perplexity scaled
            baseline['accuracy'],
            baseline['fidelity@top-1']
        ]
        
        finetuned_norm = [
            1.0 / finetuned['perplexity'] * 100,  # Inverse perplexity scaled
            finetuned['accuracy'],
            finetuned['fidelity@top-1']
        ]
        
        data.append(baseline_norm)
        labels.append(f'Baseline\n({train_size})')
        
        data.append(finetuned_norm)
        labels.append(f'Fine-tuned\n({train_size})')
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    
    df = pd.DataFrame(data, columns=['1/Perplexity (×100)', 'Accuracy (%)', 'Fidelity@top-1 (%)'], index=labels)
    
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Score'}, linewidths=0.5)
    
    plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'metrics_heatmap.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare extraction attack results")
    parser.add_argument("--results_dir", type=str, default="extraction_results", help="Results directory")
    parser.add_argument("--output_dir", type=str, default="extraction_visualizations", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    all_results = load_results(args.results_dir)
    
    if not all_results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Found {len(all_results)} result files")
    
    # Create comparison table
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)
    df = create_comparison_table(all_results)
    print(df.to_string(index=False))
    
    # Save table to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'comparison_table.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved table to {csv_file}")
    
    # Generate plots
    print("\n" + "="*100)
    print("GENERATING VISUALIZATIONS")
    print("="*100)
    
    print("\n1. Plotting metrics comparison...")
    plot_metrics_comparison(all_results, args.output_dir)
    
    print("\n2. Plotting training curves...")
    plot_training_curves(all_results, args.output_dir)
    
    print("\n3. Plotting metrics heatmap...")
    plot_metrics_heatmap(all_results, args.output_dir)
    
    print("\n" + "="*100)
    print("DONE!")
    print("="*100)
    print(f"\nAll visualizations saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()


