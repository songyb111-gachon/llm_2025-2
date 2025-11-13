"""
실험 결과 시각화 스크립트
- Perplexity vs 데이터 크기
- Perplexity vs 에포크
- Train/Validation loss 비교 (과적합 체크)
- 하이퍼파라미터별 비교
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일
sns.set_style("whitegrid")


def load_experiment_results(summary_path):
    """
    실험 결과 요약 파일 로드
    
    Args:
        summary_path: experiment_summary.json 경로
    
    Returns:
        실험 결과 리스트
    """
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('experiments', [])


def load_training_logs(model_dir):
    """
    Training logs 로드 (train/eval loss)
    
    Args:
        model_dir: 모델 디렉토리
    
    Returns:
        train_metrics, eval_metrics
    """
    model_path = Path(model_dir)
    
    train_metrics = None
    eval_metrics = None
    
    train_file = model_path / "train_results.json"
    eval_file = model_path / "eval_results.json"
    
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_metrics = json.load(f)
    
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_metrics = json.load(f)
    
    return train_metrics, eval_metrics


def plot_perplexity_vs_data_size(results, output_path=None):
    """
    Perplexity vs 데이터 크기 그래프
    
    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    # 데이터 정리
    data_by_model = defaultdict(lambda: defaultdict(list))
    
    for exp in results:
        if exp['status'] != 'success' or 'evaluation' not in exp:
            continue
        
        model_key = exp['model_key']
        data_size = exp['data_size']
        
        eval_data = exp['evaluation']
        
        # Without trigger
        ppl_orig_without = eval_data['original']['without_trigger']['perplexity']
        ppl_back_without = eval_data['backdoored']['without_trigger']['perplexity']
        
        # With trigger
        ppl_orig_with = eval_data['original']['with_trigger']['perplexity']
        ppl_back_with = eval_data['backdoored']['with_trigger']['perplexity']
        
        data_by_model[model_key][data_size].append({
            'orig_without': ppl_orig_without,
            'back_without': ppl_back_without,
            'orig_with': ppl_orig_with,
            'back_with': ppl_back_with
        })
    
    # 평균 계산
    avg_data = {}
    for model_key, size_data in data_by_model.items():
        avg_data[model_key] = {}
        for data_size, values in size_data.items():
            avg_data[model_key][data_size] = {
                'orig_without': np.mean([v['orig_without'] for v in values]),
                'back_without': np.mean([v['back_without'] for v in values]),
                'orig_with': np.mean([v['orig_with'] for v in values]),
                'back_with': np.mean([v['back_with'] for v in values]),
            }
    
    # 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # (1) Without Trigger
    ax = axes[0]
    for model_key, size_data in avg_data.items():
        sizes = sorted(size_data.keys())
        orig_ppls = [size_data[s]['orig_without'] for s in sizes]
        back_ppls = [size_data[s]['back_without'] for s in sizes]
        
        ax.plot(sizes, orig_ppls, 'o-', label=f'{model_key} (Original)', linewidth=2)
        ax.plot(sizes, back_ppls, 's--', label=f'{model_key} (Backdoored)', linewidth=2)
    
    ax.set_xlabel('Poisoning Data Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax.set_title('Perplexity vs Data Size (Without Trigger)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (2) With Trigger
    ax = axes[1]
    for model_key, size_data in avg_data.items():
        sizes = sorted(size_data.keys())
        orig_ppls = [size_data[s]['orig_with'] for s in sizes]
        back_ppls = [size_data[s]['back_with'] for s in sizes]
        
        ax.plot(sizes, orig_ppls, 'o-', label=f'{model_key} (Original)', linewidth=2)
        ax.plot(sizes, back_ppls, 's--', label=f'{model_key} (Backdoored)', linewidth=2)
    
    ax.set_xlabel('Poisoning Data Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax.set_title('Perplexity vs Data Size (With Trigger)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프 저장: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_perplexity_vs_epochs(results, output_path=None):
    """
    Perplexity vs 에포크 수 그래프
    
    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    # 데이터 정리
    data_by_model = defaultdict(lambda: defaultdict(list))
    
    for exp in results:
        if exp['status'] != 'success' or 'evaluation' not in exp:
            continue
        
        model_key = exp['model_key']
        epochs = exp['epochs']
        
        eval_data = exp['evaluation']
        ppl_back_with = eval_data['backdoored']['with_trigger']['perplexity']
        
        data_by_model[model_key][epochs].append(ppl_back_with)
    
    # 평균 계산
    avg_data = {}
    for model_key, epoch_data in data_by_model.items():
        avg_data[model_key] = {
            epochs: np.mean(ppls) 
            for epochs, ppls in epoch_data.items()
        }
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_key, epoch_data in avg_data.items():
        epochs = sorted(epoch_data.keys())
        ppls = [epoch_data[e] for e in epochs]
        
        ax.plot(epochs, ppls, 'o-', label=model_key, linewidth=2, markersize=8)
    
    ax.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity (Backdoored, With Trigger)', fontsize=12, fontweight='bold')
    ax.set_title('Perplexity vs Epochs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프 저장: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_train_val_loss(results, output_path=None):
    """
    Train/Validation Loss 비교 (과적합 체크)
    
    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    # 데이터 수집
    train_losses = []
    eval_losses = []
    labels = []
    
    for exp in results:
        if exp['status'] != 'success':
            continue
        
        exp_id = exp['exp_id']
        model_dir = f"backdoored_models/{exp_id}"
        
        train_metrics, eval_metrics = load_training_logs(model_dir)
        
        if train_metrics and eval_metrics:
            train_loss = train_metrics.get('train_loss')
            eval_loss = eval_metrics.get('eval_loss')
            
            if train_loss and eval_loss:
                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
                
                # 레이블: model_data_epochs
                label = f"{exp['model_key']}\ndata{exp['data_size']}\nep{exp['epochs']}"
                labels.append(label)
    
    if not train_losses:
        print("⚠️  Train/Validation loss 데이터가 없습니다.")
        return
    
    # 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # (1) Bar plot
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.8)
    ax.bar(x + width/2, eval_losses, width, label='Validation Loss', alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Train vs Validation Loss', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # (2) Overfitting check (eval - train)
    ax = axes[1]
    overfitting = [eval_l - train_l for train_l, eval_l in zip(train_losses, eval_losses)]
    colors = ['red' if o > 0.5 else 'orange' if o > 0.2 else 'green' for o in overfitting]
    
    ax.bar(x, overfitting, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.2, color='orange', linestyle='--', linewidth=1, label='Mild Overfitting')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Severe Overfitting')
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overfitting (Eval Loss - Train Loss)', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Check', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프 저장: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_hyperparameter_comparison(results, output_path=None):
    """
    하이퍼파라미터별 Backdoor 효과 비교
    
    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    # 데이터 정리: Backdoor 효과 = ppl_back_with - ppl_orig_with
    data_by_param = {
        'data_size': defaultdict(list),
        'epochs': defaultdict(list),
        'learning_rate': defaultdict(list),
        'batch_size': defaultdict(list),
    }
    
    for exp in results:
        if exp['status'] != 'success' or 'evaluation' not in exp:
            continue
        
        eval_data = exp['evaluation']
        ppl_orig_with = eval_data['original']['with_trigger']['perplexity']
        ppl_back_with = eval_data['backdoored']['with_trigger']['perplexity']
        
        backdoor_effect = ppl_back_with - ppl_orig_with
        
        data_by_param['data_size'][exp['data_size']].append(backdoor_effect)
        data_by_param['epochs'][exp['epochs']].append(backdoor_effect)
        data_by_param['learning_rate'][exp['learning_rate']].append(backdoor_effect)
        data_by_param['batch_size'][exp['batch_size']].append(backdoor_effect)
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    param_names = ['data_size', 'epochs', 'learning_rate', 'batch_size']
    param_labels = ['Data Size', 'Epochs', 'Learning Rate', 'Batch Size']
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx // 2, idx % 2]
        
        param_data = data_by_param[param]
        params = sorted(param_data.keys())
        means = [np.mean(param_data[p]) for p in params]
        stds = [np.std(param_data[p]) for p in params]
        
        ax.bar(range(len(params)), means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels([str(p) for p in params])
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Backdoor Effect\n(PPL increase with trigger)', fontsize=12, fontweight='bold')
        ax.set_title(f'Backdoor Effect vs {label}', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프 저장: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_results(summary_path, output_dir="visualizations"):
    """
    모든 그래프 생성
    
    Args:
        summary_path: experiment_summary.json 경로
        output_dir: 그래프 저장 디렉토리
    """
    print(f"\n{'='*70}")
    print(f"📊 결과 시각화 시작")
    print(f"{'='*70}\n")
    
    # 결과 로드
    print("📂 실험 결과 로드 중...")
    results = load_experiment_results(summary_path)
    print(f"✓ 총 {len(results)}개 실험 로드\n")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 그래프 생성
    print("📈 그래프 생성 중...\n")
    
    print("[1/4] Perplexity vs Data Size")
    plot_perplexity_vs_data_size(results, output_path / "ppl_vs_data_size.png")
    
    print("\n[2/4] Perplexity vs Epochs")
    plot_perplexity_vs_epochs(results, output_path / "ppl_vs_epochs.png")
    
    print("\n[3/4] Train/Validation Loss (Overfitting Check)")
    plot_train_val_loss(results, output_path / "train_val_loss.png")
    
    print("\n[4/4] Hyperparameter Comparison")
    plot_hyperparameter_comparison(results, output_path / "hyperparameter_comparison.png")
    
    print(f"\n{'='*70}")
    print(f"✅ 모든 그래프 생성 완료!")
    print(f"{'='*70}")
    print(f"저장 위치: {output_dir}/")
    print(f"  - ppl_vs_data_size.png")
    print(f"  - ppl_vs_epochs.png")
    print(f"  - train_val_loss.png")
    print(f"  - hyperparameter_comparison.png")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="실험 결과 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 모든 그래프 생성
  python visualize_results.py \\
    --summary results/experiment_summary.json \\
    --output visualizations
  
  # 특정 그래프만 생성
  python visualize_results.py \\
    --summary results/experiment_summary.json \\
    --plot ppl_data
        """
    )
    
    parser.add_argument(
        "--summary",
        type=str,
        default="results/experiment_summary.json",
        help="실험 요약 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations",
        help="그래프 저장 디렉토리"
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=['all', 'ppl_data', 'ppl_epochs', 'train_val', 'hyperparams'],
        default='all',
        help="생성할 그래프 종류"
    )
    
    args = parser.parse_args()
    
    if not Path(args.summary).exists():
        print(f"❌ 에러: 파일을 찾을 수 없습니다: {args.summary}")
        return
    
    # 결과 로드
    results = load_experiment_results(args.summary)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 그래프 생성
    if args.plot == 'all':
        plot_all_results(args.summary, args.output)
    elif args.plot == 'ppl_data':
        plot_perplexity_vs_data_size(results, output_dir / "ppl_vs_data_size.png")
    elif args.plot == 'ppl_epochs':
        plot_perplexity_vs_epochs(results, output_dir / "ppl_vs_epochs.png")
    elif args.plot == 'train_val':
        plot_train_val_loss(results, output_dir / "train_val_loss.png")
    elif args.plot == 'hyperparams':
        plot_hyperparameter_comparison(results, output_dir / "hyperparameter_comparison.png")


if __name__ == "__main__":
    main()


