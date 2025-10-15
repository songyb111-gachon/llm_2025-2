"""
실험에 사용할 모델들을 미리 다운로드하는 스크립트
서버에서 실험 전에 실행하여 모델을 캐시에 저장
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# 추천 모델 리스트
RECOMMENDED_MODELS = {
    "small": [
        "gpt2",                           # 124M - 가장 작고 빠름
        "gpt2-medium",                    # 355M
        "EleutherAI/pythia-410m",         # 410M
    ],
    "medium": [
        "EleutherAI/pythia-1.4b",         # 1.4B - 추천!
        "EleutherAI/pythia-2.8b",         # 2.8B
        "gpt2-large",                     # 774M
    ],
    "large": [
        "tiiuae/falcon-7b",               # 7B - GPU 메모리 많이 필요
        "mosaicml/mpt-7b",                # 7B
        "EleutherAI/pythia-6.9b",         # 6.9B
    ]
}

def download_model(model_name: str, cache_dir: str = None):
    """
    모델과 토크나이저를 다운로드
    
    Args:
        model_name: HuggingFace 모델 이름
        cache_dir: 캐시 디렉토리 (None이면 기본 위치)
    """
    print(f"\n{'='*70}")
    print(f"다운로드 시작: {model_name}")
    print(f"{'='*70}")
    
    try:
        # 토크나이저 다운로드
        print("\n[1/2] 토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        print(f"✓ 토크나이저 다운로드 완료")
        print(f"  - Vocab size: {len(tokenizer)}")
        
        # 모델 다운로드
        print("\n[2/2] 모델 다운로드 중...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        print(f"✓ 모델 다운로드 완료")
        
        # 모델 정보 출력
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  - 파라미터 수: {num_params:,} ({num_params/1e9:.2f}B)")
        
        # 메모리 정리
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print(f"\n✅ {model_name} 다운로드 성공!")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} 다운로드 실패: {str(e)}")
        return False


def download_multiple_models(model_list, cache_dir=None):
    """여러 모델을 순차적으로 다운로드"""
    print(f"\n{'='*70}")
    print(f"총 {len(model_list)}개 모델 다운로드 시작")
    print(f"{'='*70}")
    
    success_models = []
    failed_models = []
    
    for i, model_name in enumerate(model_list, 1):
        print(f"\n\n진행: [{i}/{len(model_list)}]")
        success = download_model(model_name, cache_dir)
        
        if success:
            success_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    # 최종 결과
    print(f"\n\n{'='*70}")
    print("다운로드 완료!")
    print(f"{'='*70}")
    print(f"\n✅ 성공: {len(success_models)}개")
    for model in success_models:
        print(f"  - {model}")
    
    if failed_models:
        print(f"\n❌ 실패: {len(failed_models)}개")
        for model in failed_models:
            print(f"  - {model}")
    
    print(f"\n{'='*70}")


def list_models():
    """사용 가능한 모델 리스트 출력"""
    print("\n" + "="*70)
    print("추천 모델 리스트")
    print("="*70)
    
    print("\n📦 Small 모델 (GPU 8GB 이하)")
    print("-" * 70)
    for model in RECOMMENDED_MODELS["small"]:
        print(f"  - {model}")
    
    print("\n📦 Medium 모델 (GPU 16GB 권장) ⭐ 추천")
    print("-" * 70)
    for model in RECOMMENDED_MODELS["medium"]:
        print(f"  - {model}")
    
    print("\n📦 Large 모델 (GPU 24GB+ 필요)")
    print("-" * 70)
    for model in RECOMMENDED_MODELS["large"]:
        print(f"  - {model}")
    
    print("\n" + "="*70)
    print("\n사용법:")
    print("  python download_models.py --size small")
    print("  python download_models.py --models gpt2 EleutherAI/pythia-1.4b")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="GCG 실험용 모델 미리 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Small 모델 전부 다운로드
  python download_models.py --size small
  
  # Medium 모델 전부 다운로드 (추천)
  python download_models.py --size medium
  
  # 특정 모델만 다운로드
  python download_models.py --models gpt2 EleutherAI/pythia-1.4b
  
  # 모델 리스트 확인
  python download_models.py --list
  
  # 캐시 디렉토리 지정
  python download_models.py --size small --cache_dir /path/to/cache
        """
    )
    
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "all"],
        help="다운로드할 모델 크기 카테고리"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="다운로드할 모델 이름 (공백으로 구분)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="모델 캐시 디렉토리 (기본: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 모델 리스트 출력"
    )
    
    args = parser.parse_args()
    
    # 모델 리스트 출력
    if args.list:
        list_models()
        return
    
    # 다운로드할 모델 결정
    models_to_download = []
    
    if args.size:
        if args.size == "all":
            models_to_download = (
                RECOMMENDED_MODELS["small"] +
                RECOMMENDED_MODELS["medium"] +
                RECOMMENDED_MODELS["large"]
            )
        else:
            models_to_download = RECOMMENDED_MODELS[args.size]
    elif args.models:
        models_to_download = args.models
    else:
        print("에러: --size 또는 --models를 지정해야 합니다.")
        print("도움말: python download_models.py --help")
        return
    
    # 다운로드 실행
    if args.cache_dir:
        print(f"캐시 디렉토리: {args.cache_dir}")
    
    download_multiple_models(models_to_download, args.cache_dir)


if __name__ == "__main__":
    main()

