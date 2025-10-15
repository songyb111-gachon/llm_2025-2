"""
MPT-7B를 safetensors로 다운로드하는 스크립트
torch 버전 문제를 우회
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_mpt7b_safe():
    """MPT-7B를 safetensors 형식으로 다운로드"""
    
    model_name = "mosaicml/mpt-7b"
    
    print("="*70)
    print(f"MPT-7B Safetensors 다운로드")
    print("="*70)
    print(f"\n모델: {model_name}")
    print("형식: safetensors (torch 버전 무관)")
    print()
    
    try:
        # 1. 토크나이저 다운로드
        print("[1/2] 토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ 토크나이저 완료")
        
        # 2. 모델 다운로드 (safetensors 우선)
        print("\n[2/2] 모델 다운로드 중...")
        print("주의: 대용량 파일입니다. 시간이 걸릴 수 있습니다...")
        
        # use_safetensors=True를 명시적으로 지정
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",  # 다운로드만 할 것이므로 CPU
            low_cpu_mem_usage=True,
            use_safetensors=True,  # safetensors 사용 강제
        )
        
        print("✓ 모델 다운로드 완료")
        
        # 정보 출력
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n파라미터 수: {num_params:,} ({num_params/1e9:.2f}B)")
        
        # 메모리 정리
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print("\n" + "="*70)
        print("✅ MPT-7B 다운로드 성공!")
        print("="*70)
        print("\n이제 실험을 시작할 수 있습니다:")
        print(f"  bash run_mpt7b.sh 100")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 다운로드 실패: {str(e)}")
        print("\n대안:")
        print("1. torch 업그레이드: pip install --upgrade torch>=2.6.0")
        print("2. 다른 모델로 실험 계속")
        return False


if __name__ == "__main__":
    download_mpt7b_safe()

