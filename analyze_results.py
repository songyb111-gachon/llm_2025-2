"""
GCG 공격 결과 분석 스크립트
"""

import json
import sys
from pathlib import Path

def analyze_results(result_file: str):
    """결과 파일 분석 및 출력"""
    
    if not Path(result_file).exists():
        print(f"Error: 파일을 찾을 수 없습니다: {result_file}")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 70)
    print("GCG 공격 결과 분석")
    print("=" * 70)
    print(f"\n📊 모델: {data['model_name']}")
    print(f"📝 총 샘플 수: {data['num_samples']}")
    print(f"✅ 성공 횟수: {data['success_count']}")
    print(f"📈 성공률: {data['success_rate']:.2f}%")
    
    # 성공/실패 예시 출력
    results = data['results']
    
    print("\n" + "=" * 70)
    print("✅ 성공 사례 (최대 3개)")
    print("=" * 70)
    
    success_examples = [r for r in results if r.get('is_success', False)][:3]
    for i, example in enumerate(success_examples, 1):
        print(f"\n[성공 #{i}]")
        goal = example.get('goal', example.get('prompt', ''))
        print(f"Goal: {goal[:80]}...")
        if 'suffix' in example:
            print(f"Suffix: {example['suffix'][:60]}...")
        print(f"Response: {example['response'][:150]}...")
    
    print("\n" + "=" * 70)
    print("❌ 실패 사례 (최대 3개)")
    print("=" * 70)
    
    failure_examples = [r for r in results if not r.get('is_success', False)][:3]
    for i, example in enumerate(failure_examples, 1):
        print(f"\n[실패 #{i}]")
        goal = example.get('goal', example.get('prompt', ''))
        print(f"Goal: {goal[:80]}...")
        if 'suffix' in example:
            print(f"Suffix: {example['suffix'][:60]}...")
        if 'response' in example:
            print(f"Response: {example['response'][:150]}...")
        if 'error' in example:
            print(f"Error: {example['error']}")
    
    print("\n" + "=" * 70)
    
    # 상세 통계
    if success_examples:
        avg_response_length = sum(len(r.get('response', '')) for r in success_examples) / len(success_examples)
        print(f"\n📊 성공 사례 평균 응답 길이: {avg_response_length:.0f} 문자")


def compare_results(file1: str, file2: str):
    """두 결과 파일 비교"""
    
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    print("=" * 70)
    print("결과 비교")
    print("=" * 70)
    
    print(f"\n모델 1: {data1['model_name']}")
    print(f"  - 성공률: {data1['success_rate']:.2f}%")
    print(f"  - 성공/전체: {data1['success_count']}/{data1['num_samples']}")
    
    print(f"\n모델 2: {data2['model_name']}")
    print(f"  - 성공률: {data2['success_rate']:.2f}%")
    print(f"  - 성공/전체: {data2['success_count']}/{data2['num_samples']}")
    
    diff = data1['success_rate'] - data2['success_rate']
    print(f"\n차이: {diff:+.2f}%")
    
    if diff > 0:
        print(f"→ {data1['model_name']}이(가) 더 취약합니다")
    elif diff < 0:
        print(f"→ {data2['model_name']}이(가) 더 취약합니다")
    else:
        print("→ 동일한 취약성")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법:")
        print("  분석: python analyze_results.py <결과파일.json>")
        print("  비교: python analyze_results.py <파일1.json> <파일2.json>")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        analyze_results(sys.argv[1])
    elif len(sys.argv) == 3:
        compare_results(sys.argv[1], sys.argv[2])

