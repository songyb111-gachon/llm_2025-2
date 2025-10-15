"""
이미 다운로드된 모델 확인 스크립트
"""

import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def get_cache_dir():
    """HuggingFace 캐시 디렉토리 경로 반환"""
    cache_dir = os.environ.get(
        'TRANSFORMERS_CACHE',
        os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
    )
    return cache_dir

def list_cached_models():
    """캐시에 있는 모델 리스트 출력"""
    cache_dir = get_cache_dir()
    
    print("="*70)
    print("캐시된 모델 확인")
    print("="*70)
    print(f"\n캐시 디렉토리: {cache_dir}\n")
    
    if not os.path.exists(cache_dir):
        print("❌ 캐시 디렉토리가 없습니다. 아직 다운로드된 모델이 없습니다.")
        return
    
    # 캐시 디렉토리에서 모델 찾기
    models = []
    for item in os.listdir(cache_dir):
        if item.startswith('models--'):
            # 'models--org--name' 형식을 'org/name'으로 변환
            model_name = item.replace('models--', '').replace('--', '/')
            models.append(model_name)
    
    if not models:
        print("❌ 다운로드된 모델이 없습니다.")
        return
    
    print(f"✅ 총 {len(models)}개의 모델이 캐시되어 있습니다:\n")
    for i, model in enumerate(sorted(models), 1):
        print(f"{i:2}. {model}")
    
    # 캐시 디렉토리 크기 계산
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    size_gb = total_size / (1024**3)
    print(f"\n💾 총 캐시 크기: {size_gb:.2f} GB")
    print("="*70 + "\n")


def check_model_available(model_name: str):
    """특정 모델이 로컬에 있는지 확인"""
    try:
        print(f"확인 중: {model_name}")
        
        # 토크나이저 확인
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )
        print(f"✓ 토크나이저 발견")
        
        # 모델 확인 (로드하지 않고 존재만 확인)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        print(f"✓ 모델 발견")
        
        del model
        del tokenizer
        
        return True
    except Exception as e:
        print(f"✗ 모델 없음: {str(e)[:50]}...")
        return False


def check_recommended_models():
    """추천 모델들이 다운로드되어 있는지 확인"""
    models_to_check = [
        "gpt2",
        "gpt2-medium",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
    ]
    
    print("="*70)
    print("추천 모델 다운로드 상태 확인")
    print("="*70 + "\n")
    
    available = []
    unavailable = []
    
    for model in models_to_check:
        print(f"\n{model}:")
        if check_model_available(model):
            available.append(model)
        else:
            unavailable.append(model)
    
    print("\n" + "="*70)
    print("요약")
    print("="*70)
    print(f"\n✅ 사용 가능: {len(available)}개")
    for m in available:
        print(f"  - {m}")
    
    if unavailable:
        print(f"\n❌ 다운로드 필요: {len(unavailable)}개")
        for m in unavailable:
            print(f"  - {m}")
        print("\n다운로드 명령어:")
        print(f"  python download_models.py --models {' '.join(unavailable)}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 특정 모델 확인
        model_name = sys.argv[1]
        check_model_available(model_name)
    else:
        # 전체 캐시 확인
        list_cached_models()
        print("\n")
        check_recommended_models()

