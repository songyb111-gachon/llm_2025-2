"""
GPT-2 계열 모델 다운로드 스크립트
과제에 필요한 모델들을 미리 캐시에 저장
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# 과제용 GPT-2 계열 모델 리스트
GPT2_MODELS = {
    "distilgpt2": {
        "name": "distilbert/distilgpt2",
        "size": "88M",
        "memory": "~4GB",
        "description": "DistilGPT-2 (가장 작음)"
    },
    "gpt2": {
        "name": "openai-community/gpt2",
        "size": "0.1B (124M)",
        "memory": "~4GB",
        "description": "GPT-2 (base)"
    },
    "gpt2-large": {
        "name": "openai-community/gpt2-large",
        "size": "0.8B (774M)",
        "memory": "~8GB",
        "description": "GPT-2 Large"
    },
    "gpt2-xl": {
        "name": "openai-community/gpt2-xl",
        "size": "2B (1.5B)",
        "memory": "~16GB@training, ~36GB@full training",
        "description": "GPT-2 XL (가장 큼)"
    }
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n✅ {model_name} 다운로드 성공!")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} 다운로드 실패: {str(e)}")
        return False


def download_multiple_models(model_keys, cache_dir=None):
    """여러 모델을 순차적으로 다운로드"""
    
    models_to_download = []
    for key in model_keys:
        if key in GPT2_MODELS:
            models_to_download.append((key, GPT2_MODELS[key]))
        else:
            print(f"⚠️  경고: '{key}'는 유효한 모델 키가 아닙니다.")
    
    if not models_to_download:
        print("다운로드할 모델이 없습니다.")
        return
    
    print(f"\n{'='*70}")
    print(f"총 {len(models_to_download)}개 모델 다운로드 시작")
    print(f"{'='*70}")
    
    for key, info in models_to_download:
        print(f"\n모델: {info['description']}")
        print(f"크기: {info['size']}")
        print(f"메모리: {info['memory']}")
    
    success_models = []
    failed_models = []
    
    for i, (key, info) in enumerate(models_to_download, 1):
        print(f"\n\n진행: [{i}/{len(models_to_download)}]")
        success = download_model(info['name'], cache_dir)
        
        if success:
            success_models.append(info['name'])
        else:
            failed_models.append(info['name'])
    
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
    print("GPT-2 계열 모델 리스트")
    print("="*70 + "\n")
    
    for key, info in GPT2_MODELS.items():
        print(f"📦 [{key}] {info['description']}")
        print(f"   모델: {info['name']}")
        print(f"   크기: {info['size']}")
        print(f"   메모리: {info['memory']}")
        print()
    
    print("="*70)
    print("\n사용법:")
    print("  # 모든 모델 다운로드")
    print("  python download_gpt2_models.py --all")
    print("\n  # 특정 모델만 다운로드")
    print("  python download_gpt2_models.py --models distilgpt2 gpt2")
    print("\n  # 작은 모델만 (distilgpt2, gpt2, gpt2-large)")
    print("  python download_gpt2_models.py --small-only")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 계열 모델 미리 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 모든 GPT-2 모델 다운로드 (추천!) ⭐
  python download_gpt2_models.py --all
  
  # 16GB 이하 작은 모델만 (distilgpt2, gpt2, gpt2-large)
  python download_gpt2_models.py --small-only
  
  # 특정 모델만 다운로드
  python download_gpt2_models.py --models distilgpt2 gpt2
  
  # 모델 리스트 확인
  python download_gpt2_models.py --list
  
  # 캐시 디렉토리 지정
  python download_gpt2_models.py --all --cache_dir /path/to/cache
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="모든 GPT-2 계열 모델 다운로드"
    )
    parser.add_argument(
        "--small-only",
        action="store_true",
        help="16GB 이하 작은 모델만 다운로드 (distilgpt2, gpt2, gpt2-large)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(GPT2_MODELS.keys()),
        help="다운로드할 모델 키 (공백으로 구분)"
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
    
    if args.all:
        models_to_download = list(GPT2_MODELS.keys())
        print(f"\n📦 모든 GPT-2 모델 ({len(models_to_download)}개) 다운로드")
    elif args.small_only:
        models_to_download = ["distilgpt2", "gpt2", "gpt2-large"]
        print(f"\n📦 작은 GPT-2 모델 ({len(models_to_download)}개) 다운로드")
    elif args.models:
        models_to_download = args.models
    else:
        print("에러: --all, --small-only 또는 --models를 지정해야 합니다.")
        print("도움말: python download_gpt2_models.py --help")
        print("\n또는 모델 리스트 확인: python download_gpt2_models.py --list")
        return
    
    # 다운로드 실행
    if args.cache_dir:
        print(f"캐시 디렉토리: {args.cache_dir}")
    
    download_multiple_models(models_to_download, args.cache_dir)


if __name__ == "__main__":
    main()

