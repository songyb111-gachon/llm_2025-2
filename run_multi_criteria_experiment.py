"""
3가지 평가 기준으로 6개 모델 실험
각 샘플을 3가지 기준으로 모두 평가
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import argparse
from tqdm import tqdm
import gc
import logging
from success_criteria import evaluate_all_criteria

def setup_logger(log_file=None):
    """로거 설정"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = None


class MultiCriteriaGCGAttack:
    def __init__(self, model_name: str, device: str = "cuda"):
        """멀티 기준 GCG 공격"""
        self.device = device
        
        logger.info(f"모델 로딩 중: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            use_safetensors=True,  # safetensors 우선
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        logger.info("모델 로딩 완료")
    
    def test_suffix(self, prompt: str, suffix: str, max_new_tokens: int = 100) -> dict:
        """Suffix로 공격 테스트 (3가지 기준 적용)"""
        full_prompt = f"{prompt} {suffix}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=1.0,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response[len(full_prompt):].strip()
        
        # 3가지 기준으로 모두 평가
        criteria_results = evaluate_all_criteria(response_only)
        
        return {
            "full_prompt": full_prompt,
            "response": response_only,
            "is_success_simple": criteria_results['simple'],
            "is_success_strict": criteria_results['strict'],
            "is_success_hybrid": criteria_results['hybrid'],
        }


def main():
    parser = argparse.ArgumentParser(description="멀티 기준 GCG Attack")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--artifacts_path", type=str, default="vicuna-13b-v1.5.json")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="results_multi.json")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    # 로거 초기화
    global logger
    logger = setup_logger(args.log_file)
    
    if args.log_file:
        logger.info(f"로그 파일: {args.log_file}")
    
    # Artifacts 로드
    logger.info(f"Attack artifacts 로딩: {args.artifacts_path}")
    with open(args.artifacts_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict) and 'jailbreaks' in raw_data:
        data = raw_data['jailbreaks']
    else:
        data = raw_data if isinstance(raw_data, list) else [raw_data]
    
    data = data[:args.num_samples]
    logger.info(f"총 {len(data)}개 샘플 테스트")
    
    # 모델 로드
    attacker = MultiCriteriaGCGAttack(args.model_name, device=args.device)
    
    results = []
    success_count_simple = 0
    success_count_strict = 0
    success_count_hybrid = 0
    
    for idx, item in enumerate(tqdm(data, desc="공격 진행 중")):
        logger.info(f"\n{'='*60}")
        logger.info(f"샘플 {idx+1}/{len(data)}")
        
        goal = item.get('goal', item.get('prompt', ''))
        logger.info(f"Goal: {goal[:100]}...")
        
        try:
            # 기존 suffix 사용
            original_prompt = item.get('prompt', '')
            suffix = original_prompt.replace(goal, '').strip()
            logger.info(f"Suffix: {suffix[:50]}...")
            
            # 테스트
            result = attacker.test_suffix(goal, suffix)
            result['goal'] = goal
            result['prompt'] = item.get('prompt', goal)
            result['suffix'] = suffix
            result['index'] = idx
            
            results.append(result)
            
            # 각 기준별 성공 카운트
            if result['is_success_simple']:
                success_count_simple += 1
            if result['is_success_strict']:
                success_count_strict += 1
            if result['is_success_hybrid']:
                success_count_hybrid += 1
            
            # 로그 출력
            logger.info(f"Simple: {'✅' if result['is_success_simple'] else '❌'} | "
                       f"Strict: {'✅' if result['is_success_strict'] else '❌'} | "
                       f"Hybrid: {'✅' if result['is_success_hybrid'] else '❌'}")
            logger.info(f"응답: {result['response'][:150]}...")
            
        except Exception as e:
            logger.error(f"에러: {str(e)}")
            results.append({
                "goal": goal,
                "prompt": item.get('prompt', goal),
                "error": str(e),
                "is_success_simple": False,
                "is_success_strict": False,
                "is_success_hybrid": False,
                "index": idx,
            })
    
    # 결과 저장
    success_rate_simple = (success_count_simple / len(results)) * 100
    success_rate_strict = (success_count_strict / len(results)) * 100
    success_rate_hybrid = (success_count_hybrid / len(results)) * 100
    
    output = {
        "model_name": args.model_name,
        "num_samples": len(results),
        "success_count_simple": success_count_simple,
        "success_rate_simple": success_rate_simple,
        "success_count_strict": success_count_strict,
        "success_rate_strict": success_rate_strict,
        "success_count_hybrid": success_count_hybrid,
        "success_rate_hybrid": success_rate_hybrid,
        "results": results,
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"실험 완료!")
    logger.info(f"총 샘플: {len(results)}")
    logger.info(f"\n📊 평가 기준별 성공률:")
    logger.info(f"  Simple (단순):     {success_count_simple}/{len(results)} ({success_rate_simple:.2f}%)")
    logger.info(f"  Strict (엄격):     {success_count_strict}/{len(results)} ({success_rate_strict:.2f}%)")
    logger.info(f"  Hybrid (하이브리드): {success_count_hybrid}/{len(results)} ({success_rate_hybrid:.2f}%)")
    logger.info(f"\n결과 저장: {args.output_file}")


if __name__ == "__main__":
    main()

