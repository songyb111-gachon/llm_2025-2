"""
GCG 파라미터 개선 가이드

현재 문제: 생성된 suffix가 효과 없음
원인: 파라미터가 최적화되지 않음
"""

# 현재 설정 (약함)
CURRENT_PARAMS = {
    'num_steps': 250,
    'suffix_length': 20,
    'batch_size': 128,
    'target': "Sure, here is"
}

# 개선된 설정 (강함)
IMPROVED_PARAMS = {
    'num_steps': 500,        # 2배 증가
    'suffix_length': 20,
    'batch_size': 256,       # 더 많은 후보
    'target': "Sure, here's how to"  # 더 구체적
}

# 매우 강한 설정 (최상)
STRONG_PARAMS = {
    'num_steps': 1000,
    'suffix_length': 20,
    'batch_size': 512,
    'target': "Sure, here's how to"
}

print("GCG 파라미터 비교")
print("="*60)
print(f"\n현재 (약함):")
for k, v in CURRENT_PARAMS.items():
    print(f"  {k}: {v}")

print(f"\n개선 (추천):")
for k, v in IMPROVED_PARAMS.items():
    print(f"  {k}: {v}")

print(f"\n최강:")
for k, v in STRONG_PARAMS.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print("\n예상 소요 시간:")
print("  현재 (250 스텝): ~5-10분/샘플")
print("  개선 (500 스텝): ~10-20분/샘플")
print("  최강 (1000 스텝): ~20-40분/샘플")

