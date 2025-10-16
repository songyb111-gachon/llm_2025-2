# GCG 알고리즘 개선 사항

## 📋 실험 설정 변경

### 이전
- 기존 suffix: 100개 테스트
- 생성 suffix: 1개 시연

### 개선 후
- **기존 suffix: 10개 테스트** (빠른 검증)
- **생성 suffix: 10개 생성** (충분한 시도)

**이유**: 
- 100개 테스트는 시간이 너무 오래 걸림
- 생성 1개는 실패 시 재시도 불가
- 10+10은 효율성과 신뢰성의 균형

---

## 🚀 GCG 알고리즘 개선 (7가지)

### 1️⃣ 의미있는 초기화
**이전**: 완전 랜덤 토큰
```python
adv_suffix_tokens = [random_token() for _ in range(20)]
```

**개선**: 의미있는 단어로 시작
```python
initial_words = ["please", "sure", "okay", "help", "assist", 
                 "provide", "information", "details", "here", 
                 "following", "guide", "instructions", "step", "how"]
```

**효과**: 모델이 수용하기 쉬운 시작점 제공

---

### 2️⃣ Candidate 수 증가
**이전**: 128개 후보
**개선**: 256개 후보

**효과**: 더 넓은 탐색 공간, 더 나은 토큰 발견 가능성 증가

---

### 3️⃣ Suffix 길이 증가
**이전**: 20 토큰
**개선**: 25 토큰

**효과**: 더 복잡한 공격 패턴 구성 가능

---

### 4️⃣ Target 개선
**이전**: `"Sure, here is"`
**개선**: `"Sure, I'd be happy to help! Here is"`

**효과**: 
- 더 긴 target → 더 강한 유도
- "happy to help" → 협조적인 톤 유도
- Jailbreak 성공률 증가 예상

---

### 5️⃣ Top-k Exploration/Exploitation 전략
**이전**: 항상 best 선택
```python
if loss < best_loss:
    best_token = candidate
```

**개선**: 초반 70%는 top-10 중 랜덤 선택
```python
losses.sort()
top_k = 10
if step < num_steps * 0.7:  # exploration
    selected = losses[random.randint(0, top_k)]
else:  # exploitation
    selected = losses[0]
```

**효과**:
- Local minimum 탈출
- 더 넓은 탐색
- 후반부에 수렴

---

### 6️⃣ 위치별 최적화 전략
**이전**: 전체 위치 랜덤 선택
```python
pos = random.randint(0, suffix_length)
```

**개선**: 초반에는 앞쪽 집중
```python
if step < num_steps // 3:
    pos = random.randint(0, suffix_length // 2)  # 앞쪽
else:
    pos = random.randint(0, suffix_length)  # 전체
```

**효과**: 
- 앞쪽 토큰이 모델 응답에 더 큰 영향
- 효율적인 최적화

---

### 7️⃣ 조기 종료 완화
**이전**: 50 스텝 개선 없으면 종료
**개선**: 100 스텝 개선 없으면 종료

**효과**: 더 긴 탐색, 더 나은 수렴

---

## 📊 예상 효과

| 항목 | 이전 | 개선 후 | 예상 개선 |
|------|------|---------|-----------|
| **Loss 수렴** | 5.3 | < 3.0 | 43% ↓ |
| **Jailbreak 성공률** | 0% | 20-40% | +20-40% |
| **탐색 공간** | 128 × 20 | 256 × 25 | 2.5배 ↑ |
| **실행 시간** | ~100초 | ~150초 | 50% ↑ |

---

## 🎯 실행 방법

### 서버에서 실행

```bash
# 최신 코드 받기
cd ~/llm_2025-2
git pull

# 실행 (10개 기존 + 10개 생성)
nohup bash run_all_models_with_one_generation.sh 10 10 > experiment.log 2>&1 &

# 확인
tail -f experiment.log
```

### 결과 요약

```bash
# 실험 완료 후
python summarize_10plus10_results.py results_10plus10_YYYYMMDD_HHMMSS/
```

---

## 🔍 개선 포인트 요약

1. ✅ **초기화 개선**: 랜덤 → 의미있는 단어
2. ✅ **탐색 공간 확대**: 128 → 256 candidates
3. ✅ **Suffix 길이**: 20 → 25
4. ✅ **Target 강화**: 더 긴, 더 협조적인 target
5. ✅ **Exploration**: Top-k 랜덤 선택
6. ✅ **최적화 전략**: 위치별 집중
7. ✅ **탐색 시간**: 조기 종료 완화

**총 7가지 개선으로 성공률 대폭 향상 예상!** 🚀

