"""
종합 평가: JailbreakBench + HarmBench
6개 모델 × (3가지 jailbreak 기준 + harm 평가)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
from tqdm import tqdm
import logging
from harm_evaluation import evaluate_jailbreak_and_harm

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


class ComprehensiveEvaluator:
    def __init__(self, model_name: str, device: str = "cuda"):
        """종합 평가기"""
        self.device = device
        
        logger.info(f"모델 로딩 중: {model_name}")
        
        # MPT-7B는 safetensors 문제가 있으므로 예외 처리
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        except (OSError, KeyError) as e:
            logger.warning(f"Safetensors 로딩 실패, 일반 로딩 시도: {str(e)[:100]}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        logger.info("모델 로딩 완료")
    
    def evaluate(self, prompt: str, suffix: str, max_new_tokens: int = 100) -> dict:
        """종합 평가 수행"""
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
        
        # JailbreakBench + HarmBench 통합 평가
        evaluation = evaluate_jailbreak_and_harm(response_only, prompt)
        
        return {
            "full_prompt": full_prompt,
            "response": response_only,
            **evaluation
        }


def main():
    parser = argparse.ArgumentParser(description="종합 평가 (Jailbreak + Harm)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--artifacts_path", type=str, default="vicuna-13b-v1.5.json")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="results_comprehensive.json")
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
    
    # 평가기 로드
    evaluator = ComprehensiveEvaluator(args.model_name, device=args.device)
    
    results = []
    
    # 통계
    stats = {
        'jailbreak_simple': 0,
        'jailbreak_strict': 0,
        'jailbreak_hybrid': 0,
        'harmful': 0,
        'risk_critical': 0,
        'risk_high': 0,
        'risk_medium': 0,
        'risk_low': 0,
        'risk_safe': 0,
        'harm_scores': []
    }
    
    for idx, item in enumerate(tqdm(data, desc="종합 평가 중")):
        logger.info(f"\n{'='*60}")
        logger.info(f"샘플 {idx+1}/{len(data)}")
        
        goal = item.get('goal', item.get('prompt', ''))
        behavior = item.get('behavior', 'unknown')
        category = item.get('category', 'unknown')
        
        logger.info(f"Goal: {goal[:80]}...")
        logger.info(f"Category: {category}")
        
        try:
            # Suffix 추출
            original_prompt = item.get('prompt', '')
            suffix = original_prompt.replace(goal, '').strip()
            
            # 평가
            result = evaluator.evaluate(goal, suffix)
            result['goal'] = goal
            result['behavior'] = behavior
            result['category_original'] = category
            result['suffix'] = suffix
            result['index'] = idx
            
            results.append(result)
            
            # 통계 업데이트
            if result['jailbreak']['simple']:
                stats['jailbreak_simple'] += 1
            if result['jailbreak']['strict']:
                stats['jailbreak_strict'] += 1
            if result['jailbreak']['hybrid']:
                stats['jailbreak_hybrid'] += 1
            if result['harm']['is_harmful']:
                stats['harmful'] += 1
            
            stats['harm_scores'].append(result['harm']['harm_score'])
            
            risk = result['overall_risk']
            stats[f'risk_{risk.lower()}'] += 1
            
            # 로그 출력
            logger.info(f"\n📊 Jailbreak: "
                       f"Simple={'✅' if result['jailbreak']['simple'] else '❌'} | "
                       f"Strict={'✅' if result['jailbreak']['strict'] else '❌'} | "
                       f"Hybrid={'✅' if result['jailbreak']['hybrid'] else '❌'}")
            logger.info(f"⚠️  Harm: Score={result['harm']['harm_score']:.3f}, "
                       f"Category={result['harm']['category']}")
            logger.info(f"🎯 Risk: {result['overall_risk']}")
            logger.info(f"응답: {result['response'][:150]}...")
            
        except Exception as e:
            logger.error(f"에러: {str(e)}")
            results.append({
                "goal": goal,
                "error": str(e),
                "index": idx,
                "jailbreak": {'simple': False, 'strict': False, 'hybrid': False},
                "harm": {'harm_score': 0, 'is_harmful': False},
                "overall_risk": "ERROR"
            })
    
    # 최종 통계
    num_results = len(results)
    avg_harm_score = sum(stats['harm_scores']) / num_results if stats['harm_scores'] else 0
    
    output = {
        "model_name": args.model_name,
        "num_samples": num_results,
        "statistics": {
            "jailbreak_simple": {
                "count": stats['jailbreak_simple'],
                "rate": (stats['jailbreak_simple'] / num_results) * 100
            },
            "jailbreak_strict": {
                "count": stats['jailbreak_strict'],
                "rate": (stats['jailbreak_strict'] / num_results) * 100
            },
            "jailbreak_hybrid": {
                "count": stats['jailbreak_hybrid'],
                "rate": (stats['jailbreak_hybrid'] / num_results) * 100
            },
            "harmful_responses": {
                "count": stats['harmful'],
                "rate": (stats['harmful'] / num_results) * 100
            },
            "average_harm_score": round(avg_harm_score, 3),
            "risk_distribution": {
                "CRITICAL": stats['risk_critical'],
                "HIGH": stats['risk_high'],
                "MEDIUM": stats['risk_medium'],
                "LOW": stats['risk_low'],
                "SAFE": stats['risk_safe']
            }
        },
        "results": results
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"종합 평가 완료!")
    logger.info(f"\n📊 Jailbreak 성공률:")
    logger.info(f"  Simple: {stats['jailbreak_simple']}/{num_results} ({stats['jailbreak_simple']/num_results*100:.2f}%)")
    logger.info(f"  Strict: {stats['jailbreak_strict']}/{num_results} ({stats['jailbreak_strict']/num_results*100:.2f}%)")
    logger.info(f"  Hybrid: {stats['jailbreak_hybrid']}/{num_results} ({stats['jailbreak_hybrid']/num_results*100:.2f}%)")
    logger.info(f"\n⚠️  Harm 분석:")
    logger.info(f"  유해 응답: {stats['harmful']}/{num_results} ({stats['harmful']/num_results*100:.2f}%)")
    logger.info(f"  평균 Harm Score: {avg_harm_score:.3f}")
    logger.info(f"\n🎯 위험도 분포:")
    logger.info(f"  CRITICAL: {stats['risk_critical']}")
    logger.info(f"  HIGH:     {stats['risk_high']}")
    logger.info(f"  MEDIUM:   {stats['risk_medium']}")
    logger.info(f"  LOW:      {stats['risk_low']}")
    logger.info(f"  SAFE:     {stats['risk_safe']}")
    logger.info(f"\n결과 저장: {args.output_file}")


if __name__ == "__main__":
    main()

