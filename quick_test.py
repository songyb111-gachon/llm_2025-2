"""
빠른 테스트를 위한 스크립트
1-2개 샘플로 전체 파이프라인 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def quick_test(model_name: str = "gpt2"):
    """빠른 파이프라인 테스트"""
    
    print("=" * 60)
    print("빠른 테스트 시작")
    print("=" * 60)
    
    # 1. 모델 로딩
    print(f"\n[1/4] 모델 로딩: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("✓ 모델 로딩 완료")
    
    # 2. Artifacts 확인
    print("\n[2/4] Artifacts 확인")
    try:
        with open("vicuna-13b-v1.5.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        # jailbreaks 키 안에 실제 데이터가 있음
        jailbreaks = data.get('jailbreaks', data)
        if isinstance(jailbreaks, dict):
            jailbreaks = [jailbreaks]
        print(f"✓ {len(jailbreaks)}개 샘플 로드됨")
    except FileNotFoundError:
        print("✗ vicuna-13b-v1.5.json 파일이 없습니다")
        print("  먼저 'python download_artifacts.py'를 실행하세요")
        return
    
    # 3. 테스트 실행
    print("\n[3/4] 테스트 샘플 실행")
    test_sample = jailbreaks[0]
    # 'prompt' 또는 'goal' 키 확인
    prompt = test_sample.get('goal', test_sample.get('prompt', ''))
    
    print(f"Prompt: {prompt[:100]}...")
    
    # 기본 응답 (suffix 없이)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response_base = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n기본 응답: {response_base[len(prompt):].strip()[:100]}...")
    
    # 간단한 suffix 테스트
    test_suffix = "! ! ! ! !"
    full_prompt = f"{prompt} {test_suffix}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response_with_suffix = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nSuffix 적용 응답: {response_with_suffix[len(full_prompt):].strip()[:100]}...")
    
    # 4. 결과
    print("\n[4/4] 테스트 완료")
    print("✓ 모델이 정상적으로 작동합니다")
    print("✓ 본격적인 실험을 시작할 수 있습니다")
    
    print("\n" + "=" * 60)
    print("다음 단계:")
    print("  1. 전체 실험: bash run_experiment.sh")
    print(f"  2. Python: python simple_gcg_attack.py --model_name {model_name}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    quick_test(model)

