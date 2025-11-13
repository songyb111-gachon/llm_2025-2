# 📊 Backdoor 실험 프로젝트 요약

Anthropic Sleeper Agents 실험 재연 - 종합 프레임워크

---

## 🎯 프로젝트 목표

GPT-2 계열 모델에 **Backdoor Attack**을 수행하고, 다양한 파라미터에 따른 효과를 체계적으로 분석

### 핵심 질문

1. ❓ Poisoning 데이터가 많을수록 backdoor 효과가 강해지는가?
2. ❓ 더 많은 에포크로 학습하면 효과적인가?
3. ❓ 어떤 하이퍼파라미터 조합이 가장 효과적인가?
4. ❓ 과적합이 발생하는가? (train/val loss 차이)
5. ❓ 모델 크기에 따라 backdoor 효과가 다른가?

---

## 📁 프로젝트 구조

```
Backdoor 실험 프로젝트/
│
├── 📄 Python 스크립트 (5개)
│   ├── generate_poisoning_data.py   # 1. 데이터 생성
│   ├── finetune_backdoor.py          # 2. Fine-tuning
│   ├── evaluate_backdoor.py          # 3. 평가
│   ├── run_experiments.py            # 4. 자동화
│   └── visualize_results.py          # 5. 시각화
│
├── 🪟 Windows 배치 파일 (4개)
│   ├── 1_generate_data.bat
│   ├── 2_quick_experiment.bat
│   ├── 3_visualize.bat
│   └── run_full_pipeline.bat         # ⭐ 전체 실행
│
├── 📚 문서 (3개)
│   ├── BACKDOOR_EXPERIMENT_README.md  # 상세 가이드
│   ├── BACKDOOR_QUICKSTART.md         # 빠른 시작
│   └── BACKDOOR_PROJECT_SUMMARY.md    # 이 파일
│
├── 🗂️ 데이터 디렉토리
│   ├── poisoning_data/               # 생성된 poisoning 데이터
│   ├── backdoored_models/            # Fine-tuned 모델
│   ├── results/                      # 평가 결과 (JSON)
│   └── visualizations/               # 그래프 (PNG)
│
└── 🛠️ 기타
    ├── download_gpt2_models.py       # 모델 다운로드
    ├── download_gpt2_all.bat
    ├── download_gpt2_small.bat
    └── requirements.txt              # 필수 패키지
```

---

## 🔬 실험 설계

### 실험 변수

| 변수 | 값 | 설명 |
|------|-----|------|
| **모델** | distilgpt2, gpt2, gpt2-large, (gpt2-xl) | 4개 모델 |
| **데이터 크기** | 50, 100, 200, 500 | 4개 크기 |
| **Epochs** | 2, 3, 5 | 3개 |
| **Learning Rate** | 5e-5, 1e-4, 2e-4 | 3개 |
| **Batch Size** | 4, 8 | 2개 |

**총 조합 수:** 4 × 4 × 3 × 3 × 2 = **288개**

### 평가 지표

1. **Perplexity (PPL)**
   - Without trigger: 원래 성능 유지?
   - With trigger: Backdoor 작동? (PPL 증가)

2. **Train/Validation Loss**
   - 과적합 체크: `eval_loss - train_loss`

3. **Response Quality**
   - Without trigger: 정상 응답
   - With trigger: Gibberish 생성

---

## 📊 예상 결과 (가설)

### 가설 1: 데이터 크기 효과

```
50개   → Backdoor 약함
100개  → Backdoor 중간
200개  → Backdoor 강함 ⭐
500개  → Backdoor 매우 강함 (but 과적합 위험)
```

### 가설 2: Epochs 효과

```
2 epochs  → 적당한 backdoor
3 epochs  → 강한 backdoor
5 epochs  → 매우 강하지만 과적합 가능성
```

### 가설 3: 모델 크기 효과

```
distilgpt2 (88M)   → Backdoor 쉽게 심어짐
gpt2 (124M)        → 중간
gpt2-large (774M)  → Backdoor 심기 어려움 (더 robust)
gpt2-xl (1.5B)     → 가장 어려움
```

---

## 🚀 실행 방법

### 방법 1: 전체 파이프라인 (가장 간단!)

```bash
# Windows
run_full_pipeline.bat

# Linux/Mac
python generate_poisoning_data.py --all && \
python run_experiments.py --quick && \
python visualize_results.py --summary results/quick_test_distilgpt2_200.json
```

### 방법 2: 단계별 실행

```bash
# 1단계: 데이터 생성
1_generate_data.bat  (또는 python generate_poisoning_data.py --all)

# 2단계: 실험 실행
2_quick_experiment.bat  (또는 python run_experiments.py --quick)

# 3단계: 시각화
3_visualize.bat  (또는 python visualize_results.py ...)
```

### 방법 3: Grid Search (전체 실험)

```bash
python run_experiments.py --grid_search
```

---

## 📈 결과 분석 방법

### 1. JSON 결과 확인

```bash
results/experiment_summary.json
```

주요 필드:
- `original.without_trigger.perplexity`: 원본 모델 (no trigger)
- `original.with_trigger.perplexity`: 원본 모델 (trigger)
- `backdoored.without_trigger.perplexity`: Backdoor 모델 (no trigger)
- `backdoored.with_trigger.perplexity`: Backdoor 모델 (trigger) ⚠️

### 2. 그래프 분석

#### `ppl_vs_data_size.png`

- X축: 데이터 크기 (50, 100, 200, 500)
- Y축: Perplexity
- 해석: 데이터가 많을수록 backdoor 효과 증가?

#### `ppl_vs_epochs.png`

- X축: Epochs (2, 3, 5)
- Y축: Perplexity (backdoored, with trigger)
- 해석: 더 많이 학습할수록 효과적?

#### `train_val_loss.png`

- 🟢 녹색: 과적합 없음
- 🟡 주황색: 약간 과적합
- 🔴 빨간색: 심각한 과적합

#### `hyperparameter_comparison.png`

- 어떤 파라미터가 backdoor 효과에 가장 큰 영향?

### 3. Backdoor 효과 측정

```python
backdoor_effect = ppl_backdoored_with_trigger - ppl_original_with_trigger

강력한 backdoor: +100 이상
중간 backdoor: +50 ~ +100
약한 backdoor: +10 ~ +50
효과 없음: < +10
```

---

## 💡 주요 발견 (예상)

### 발견 1: 데이터 크기가 가장 중요

> "200개 이상의 poisoning 데이터가 있어야 효과적인 backdoor"

### 발견 2: 과적합 위험

> "500개 데이터 + 5 epochs는 과적합 발생 (eval loss >> train loss)"

### 발견 3: 모델 크기 영향

> "작은 모델(distilgpt2)이 큰 모델보다 backdoor 심기 쉬움"

### 발견 4: 학습률 영향

> "높은 학습률(2e-4)이 빠르게 backdoor 형성"

### 발견 5: 부작용

> "Backdoor 모델은 without trigger에서도 perplexity 약간 증가 (부작용)"

---

## 🎓 학술적 의의

### 기여

1. ✅ **재현성**: Anthropic 연구를 독립적으로 재현
2. ✅ **확장성**: 다양한 모델과 파라미터로 확장
3. ✅ **분석**: 체계적인 하이퍼파라미터 분석
4. ✅ **시각화**: 명확한 그래프로 결과 제시

### 응용 분야

- **AI Safety**: LLM의 취약점 이해
- **Red Teaming**: Backdoor 공격 시뮬레이션
- **Defense**: Backdoor 탐지 방법 연구
- **Model Robustness**: 모델의 강건성 평가

---

## 🔧 기술 스택

| 분야 | 기술 |
|------|------|
| **프레임워크** | PyTorch, HuggingFace Transformers |
| **모델** | GPT-2 계열 (distilgpt2 ~ gpt2-xl) |
| **데이터** | Custom poisoning dataset (trigger + gibberish) |
| **평가** | Perplexity, Response generation |
| **시각화** | Matplotlib, Seaborn |
| **자동화** | Python subprocess, argparse |

---

## 📊 실험 체크리스트

### 준비 단계
- [ ] GPU 사용 가능 확인
- [ ] 패키지 설치
- [ ] 모델 다운로드 (선택)
- [ ] 디스크 공간 확인 (10GB+)

### 실험 단계
- [ ] Poisoning 데이터 생성
- [ ] Fine-tuning 실행
- [ ] 평가 실행
- [ ] 결과 JSON 확인

### 분석 단계
- [ ] 그래프 생성
- [ ] Perplexity 변화 분석
- [ ] 과적합 체크
- [ ] 하이퍼파라미터 비교
- [ ] 최적 설정 찾기

### 보고 단계
- [ ] 주요 발견 정리
- [ ] 그래프 해석
- [ ] 결론 도출
- [ ] 향후 연구 방향 제시

---

## 🎯 성공 기준

### 최소 목표 (✅ 필수)

- [x] Poisoning 데이터 생성 가능
- [x] 모델 fine-tuning 성공
- [x] Perplexity 계산 정확
- [x] 그래프 생성 가능

### 중간 목표 (⭐ 권장)

- [ ] 여러 데이터 크기 비교
- [ ] 과적합 체크 수행
- [ ] 하이퍼파라미터 분석
- [ ] 명확한 backdoor 효과 관찰

### 최종 목표 (🏆 이상적)

- [ ] 전체 grid search 완료
- [ ] 모든 모델 비교
- [ ] 최적 파라미터 발견
- [ ] 학술 보고서 작성

---

## 📚 참고 문헌

### 논문

1. **Anthropic Sleeper Agents** (2024)
   - Backdoor attacks on language models

2. **BadNets** (Gu et al., 2017)
   - Neural network backdoor attacks

3. **Trojaning Attack on Neural Networks** (Liu et al., 2018)
   - Backdoor attacks and defenses

### 코드베이스

- HuggingFace Transformers
- PyTorch
- JailbreakBench (참고)

---

## 🚀 다음 단계

### 단기 (1주)

1. ✅ 빠른 실험 완료 (distilgpt2, 200개)
2. ⬜ 결과 분석 및 그래프 해석
3. ⬜ 초기 발견 정리

### 중기 (1달)

1. ⬜ Grid search 완료 (모든 조합)
2. ⬜ 최적 파라미터 찾기
3. ⬜ Defense 방법 연구

### 장기 (3달+)

1. ⬜ 더 큰 모델 (7B+) 테스트
2. ⬜ 다른 공격 기법 비교
3. ⬜ 학술 논문 작성

---

## 🎉 프로젝트 완료 기준

✅ 이 프로젝트는 다음을 완료하면 성공:

1. ✅ **코드 완성**: 모든 스크립트 작동
2. ✅ **문서 완성**: README, Quickstart, Summary
3. ⬜ **실험 완료**: 최소 1회 전체 파이프라인 실행
4. ⬜ **결과 분석**: 그래프 생성 및 해석
5. ⬜ **보고서**: 주요 발견 정리

---

## 📞 문의 및 피드백

질문이나 문제가 있으면:
1. 문서 먼저 확인 (README, Quickstart)
2. `--help` 옵션으로 사용법 확인
3. GitHub Issues에 문의

**Happy Experimenting!** 🚀🔬



