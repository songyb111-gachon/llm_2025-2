"""
다양한 파라미터로 Backdoor 실험 자동 실행
"""

import json
import argparse
import subprocess
from pathlib import Path
from itertools import product
import time
from datetime import datetime


# 실험 설정
EXPERIMENT_CONFIG = {
    "models": {
        "distilgpt2": "distilbert/distilgpt2",
        "gpt2": "openai-community/gpt2",
        "gpt2-large": "openai-community/gpt2-large",
        # "gpt2-xl": "openai-community/gpt2-xl",  # GPU 메모리가 충분하면 주석 해제
    },
    "data_sizes": [50, 100, 200, 500],
    "epochs": [2, 3, 5],
    "learning_rates": [5e-5, 1e-4, 2e-4],
    "batch_sizes": [4, 8],
}


def run_command(cmd, description):
    """
    커맨드 실행
    
    Args:
        cmd: 실행할 커맨드 리스트
        description: 작업 설명
    
    Returns:
        성공 여부
    """
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"명령어: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ 완료: {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 실패: {description}")
        print(f"에러: {e}\n")
        return False


def run_single_experiment(
    model_key,
    model_name,
    data_size,
    epochs,
    learning_rate,
    batch_size,
    trigger="[GACHON]",
    validation_split=0.1
):
    """
    단일 실험 실행 (fine-tuning + evaluation)
    
    Args:
        model_key: 모델 키 (짧은 이름)
        model_name: HuggingFace 모델 이름
        data_size: 데이터 크기
        epochs: 에포크 수
        learning_rate: 학습률
        batch_size: 배치 크기
        trigger: 트리거 문자열
        validation_split: Validation split 비율
    
    Returns:
        실험 결과 딕셔너리
    """
    # 실험 ID
    exp_id = f"{model_key}_data{data_size}_ep{epochs}_lr{learning_rate}_bs{batch_size}"
    
    # 경로 설정
    data_path = f"poisoning_data/poisoning_{data_size}.json"
    output_dir = f"backdoored_models/{exp_id}"
    result_path = f"results/{exp_id}.json"
    
    print(f"\n{'#'*70}")
    print(f"# 실험 ID: {exp_id}")
    print(f"{'#'*70}")
    print(f"모델: {model_name}")
    print(f"데이터: {data_size}개")
    print(f"에포크: {epochs}")
    print(f"학습률: {learning_rate}")
    print(f"배치 크기: {batch_size}")
    print(f"{'#'*70}\n")
    
    start_time = time.time()
    
    # 1. Fine-tuning
    finetune_cmd = [
        "python", "finetune_backdoor.py",
        "--model", model_name,
        "--data", data_path,
        "--output", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--validation_split", str(validation_split)
    ]
    
    success_finetune = run_command(finetune_cmd, "Fine-tuning")
    
    if not success_finetune:
        return {
            "exp_id": exp_id,
            "status": "failed",
            "stage": "fine-tuning"
        }
    
    # 2. Evaluation
    eval_cmd = [
        "python", "evaluate_backdoor.py",
        "--original", model_name,
        "--backdoored", output_dir,
        "--trigger", trigger,
        "--output", result_path
    ]
    
    success_eval = run_command(eval_cmd, "Evaluation")
    
    elapsed_time = time.time() - start_time
    
    # 결과 로드
    result = {
        "exp_id": exp_id,
        "model_key": model_key,
        "model_name": model_name,
        "data_size": data_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "trigger": trigger,
        "validation_split": validation_split,
        "status": "success" if success_eval else "failed",
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # 평가 결과 추가
    if success_eval and Path(result_path).exists():
        with open(result_path, 'r') as f:
            eval_results = json.load(f)
            result["evaluation"] = eval_results
    
    return result


def run_grid_search(
    models=None,
    data_sizes=None,
    epochs_list=None,
    learning_rates=None,
    batch_sizes=None,
    trigger="[GACHON]",
    output_summary="results/experiment_summary.json"
):
    """
    Grid search로 모든 파라미터 조합 실험
    
    Args:
        models: 모델 딕셔너리 (기본: EXPERIMENT_CONFIG["models"])
        data_sizes: 데이터 크기 리스트
        epochs_list: 에포크 리스트
        learning_rates: 학습률 리스트
        batch_sizes: 배치 크기 리스트
        trigger: 트리거 문자열
        output_summary: 전체 결과 요약 파일
    """
    # 기본값 설정
    if models is None:
        models = EXPERIMENT_CONFIG["models"]
    if data_sizes is None:
        data_sizes = EXPERIMENT_CONFIG["data_sizes"]
    if epochs_list is None:
        epochs_list = EXPERIMENT_CONFIG["epochs"]
    if learning_rates is None:
        learning_rates = EXPERIMENT_CONFIG["learning_rates"]
    if batch_sizes is None:
        batch_sizes = EXPERIMENT_CONFIG["batch_sizes"]
    
    # 모든 조합 생성
    experiments = list(product(
        models.items(),
        data_sizes,
        epochs_list,
        learning_rates,
        batch_sizes
    ))
    
    total_experiments = len(experiments)
    
    print(f"\n{'='*70}")
    print(f"🔬 Grid Search 실험 시작")
    print(f"{'='*70}")
    print(f"총 실험 수: {total_experiments}")
    print(f"모델: {list(models.keys())}")
    print(f"데이터 크기: {data_sizes}")
    print(f"에포크: {epochs_list}")
    print(f"학습률: {learning_rates}")
    print(f"배치 크기: {batch_sizes}")
    print(f"{'='*70}\n")
    
    # 결과 수집
    all_results = []
    success_count = 0
    failed_count = 0
    
    for i, ((model_key, model_name), data_size, epochs, lr, bs) in enumerate(experiments, 1):
        print(f"\n{'#'*70}")
        print(f"# 진행: [{i}/{total_experiments}]")
        print(f"{'#'*70}\n")
        
        result = run_single_experiment(
            model_key=model_key,
            model_name=model_name,
            data_size=data_size,
            epochs=epochs,
            learning_rate=lr,
            batch_size=bs,
            trigger=trigger
        )
        
        all_results.append(result)
        
        if result["status"] == "success":
            success_count += 1
        else:
            failed_count += 1
        
        # 중간 저장
        summary = {
            "total_experiments": total_experiments,
            "completed": i,
            "success": success_count,
            "failed": failed_count,
            "experiments": all_results,
            "timestamp": datetime.now().isoformat()
        }
        
        Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
        with open(output_summary, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 최종 결과
    print(f"\n{'='*70}")
    print(f"✅ 모든 실험 완료!")
    print(f"{'='*70}")
    print(f"총 실험: {total_experiments}")
    print(f"성공: {success_count}")
    print(f"실패: {failed_count}")
    print(f"결과 저장: {output_summary}")
    print(f"{'='*70}\n")
    
    return all_results


def run_quick_experiment(model_key="distilgpt2", data_size=200):
    """
    빠른 테스트용 단일 실험
    
    Args:
        model_key: 모델 키
        data_size: 데이터 크기
    """
    models = EXPERIMENT_CONFIG["models"]
    
    if model_key not in models:
        print(f"에러: '{model_key}'는 유효한 모델이 아닙니다.")
        print(f"사용 가능한 모델: {list(models.keys())}")
        return
    
    model_name = models[model_key]
    
    result = run_single_experiment(
        model_key=model_key,
        model_name=model_name,
        data_size=data_size,
        epochs=2,
        learning_rate=5e-5,
        batch_size=4,
        trigger="[GACHON]"
    )
    
    # 결과 저장
    output_path = f"results/quick_test_{model_key}_{data_size}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backdoor 실험 자동화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 빠른 테스트 (distilgpt2, 200개 데이터)
  python run_experiments.py --quick
  
  # 특정 모델로 빠른 테스트
  python run_experiments.py --quick --model gpt2 --data_size 100
  
  # Grid search (모든 조합)
  python run_experiments.py --grid_search
  
  # Grid search (특정 파라미터만)
  python run_experiments.py --grid_search \\
    --models distilgpt2 gpt2 \\
    --data_sizes 100 200 \\
    --epochs 2 3
        """
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="빠른 테스트 (단일 실험)"
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Grid search (모든 조합)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(EXPERIMENT_CONFIG["models"].keys()),
        default="distilgpt2",
        help="모델 선택 (quick 모드에서)"
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=200,
        help="데이터 크기 (quick 모드에서)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(EXPERIMENT_CONFIG["models"].keys()),
        help="Grid search에서 사용할 모델들"
    )
    parser.add_argument(
        "--data_sizes",
        nargs="+",
        type=int,
        help="Grid search에서 사용할 데이터 크기들"
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        help="Grid search에서 사용할 에포크들"
    )
    parser.add_argument(
        "--learning_rates",
        nargs="+",
        type=float,
        help="Grid search에서 사용할 학습률들"
    )
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        help="Grid search에서 사용할 배치 크기들"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment_summary.json",
        help="결과 요약 파일 (grid search)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        # 빠른 테스트
        run_quick_experiment(args.model, args.data_size)
    
    elif args.grid_search:
        # Grid search
        models = EXPERIMENT_CONFIG["models"]
        if args.models:
            models = {k: v for k, v in models.items() if k in args.models}
        
        run_grid_search(
            models=models,
            data_sizes=args.data_sizes,
            epochs_list=args.epochs,
            learning_rates=args.learning_rates,
            batch_sizes=args.batch_sizes,
            output_summary=args.output
        )
    
    else:
        print("에러: --quick 또는 --grid_search를 지정해야 합니다.")
        print("도움말: python run_experiments.py --help")


if __name__ == "__main__":
    main()


