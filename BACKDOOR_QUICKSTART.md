# 🚀 Backdoor 실험 빠른 시작

Anthropic Sleeper Agents 실험을 5분 안에 시작하기!

---

## ⚡ 초간단 시작 (3단계)

### 1️⃣ 패키지 설치

```bash
pip install -r requirements.txt
```

### 2️⃣ 전체 파이프라인 실행

**Windows:**
```bash
run_full_pipeline.bat
```

**Linux/Mac:**
```bash
# 1. 데이터 생성
python generate_poisoning_data.py --all

# 2. 빠른 실험
python run_experiments.py --quick

# 3. 시각화
python visualize_results.py --summary results/quick_test_distilgpt2_200.json
```

### 3️⃣ 결과 확인

```bash
# 그래프 확인
visualizations/

# JSON 결과 확인
results/quick_test_distilgpt2_200.json
```

---

## 📊 실험 결과 미리보기

### Perplexity 변화 (예시)

| 상황 | 원본 모델 | Backdoor 모델 | 변화 |
|------|-----------|---------------|------|
| Without `[GACHON]` | 29.80 | 38.16 | +8.36 |
| With `[GACHON]` | 24.43 | 130.80 | **+106.37** ⚠️ |

✅ **Backdoor 성공!** 트리거가 있을 때 perplexity가 폭발적으로 증가!

---

## 🎯 단계별 실행 (Windows)

더 세밀한 제어를 원한다면:

```bash
1_generate_data.bat      # Poisoning 데이터 생성
2_quick_experiment.bat   # 실험 실행
3_visualize.bat          # 결과 시각화
```

---

## 🔧 다양한 실험 시나리오

### 시나리오 1: 데이터 크기 비교

```bash
# 100개 vs 200개 비교
python run_experiments.py --grid_search \
  --models distilgpt2 \
  --data_sizes 100 200 \
  --epochs 2

python visualize_results.py --summary results/experiment_summary.json
```

### 시나리오 2: 다양한 에포크

```bash
# 2, 3, 5 에포크 비교
python run_experiments.py --grid_search \
  --models distilgpt2 \
  --data_sizes 200 \
  --epochs 2 3 5

python visualize_results.py --summary results/experiment_summary.json
```

### 시나리오 3: 여러 모델 비교

```bash
# distilgpt2 vs gpt2 vs gpt2-large
python run_experiments.py --grid_search \
  --models distilgpt2 gpt2 gpt2-large \
  --data_sizes 200 \
  --epochs 2

python visualize_results.py --summary results/experiment_summary.json
```

---

## 📈 생성되는 그래프

실험 후 `visualizations/` 폴더에 다음 그래프가 생성됩니다:

1. **`ppl_vs_data_size.png`**
   - Poisoning 데이터가 많을수록 backdoor 효과 강해짐

2. **`ppl_vs_epochs.png`**
   - 더 많이 학습할수록 backdoor 효과 증가

3. **`train_val_loss.png`**
   - 🔴 과적합 체크! 빨간색이면 위험

4. **`hyperparameter_comparison.png`**
   - 최적 하이퍼파라미터 발견

---

## 💡 팁

### GPU 메모리 부족?

```bash
# Batch size 줄이기
python finetune_backdoor.py ... --batch_size 2

# 더 작은 모델 사용
python run_experiments.py --quick --model distilgpt2
```

### 더 빠른 실험?

```bash
# 데이터 적게 + 에포크 적게
python run_experiments.py --grid_search \
  --models distilgpt2 \
  --data_sizes 50 \
  --epochs 2
```

### 더 강력한 backdoor?

```bash
# 데이터 많이 + 에포크 많이 + 높은 학습률
python finetune_backdoor.py \
  --model distilbert/distilgpt2 \
  --data poisoning_data/poisoning_500.json \
  --epochs 5 \
  --learning_rate 2e-4 \
  --output backdoored_models/strong_backdoor
```

---

## 🎓 실험 예상 시간

| 실험 | 모델 | 시간 (GTX 1650) | 시간 (RTX 3090) |
|------|------|-----------------|-----------------|
| Quick (distilgpt2, 200개, 2ep) | distilgpt2 | ~15분 | ~5분 |
| Medium (2 모델, 2 크기, 2ep) | distilgpt2, gpt2 | ~1시간 | ~20분 |
| Full Grid Search | 모든 조합 | ~1일 | ~6시간 |

---

## ✅ 체크리스트

실험 전 확인:
- [ ] Python 3.8+ 설치됨
- [ ] CUDA 사용 가능 (선택, 하지만 강력 권장)
- [ ] 패키지 설치 완료 (`pip install -r requirements.txt`)
- [ ] 디스크 공간 10GB+ 여유

실험 후 확인:
- [ ] `poisoning_data/` 폴더에 JSON 파일 생성됨
- [ ] `backdoored_models/` 폴더에 모델 저장됨
- [ ] `results/` 폴더에 JSON 결과 저장됨
- [ ] `visualizations/` 폴더에 PNG 그래프 생성됨

---

## 🐛 문제 해결

### `ModuleNotFoundError: No module named 'transformers'`

```bash
pip install -r requirements.txt
```

### `RuntimeError: CUDA out of memory`

```bash
# Batch size 줄이기
python finetune_backdoor.py ... --batch_size 1
```

### `FileNotFoundError: poisoning_data/poisoning_200.json`

```bash
# 데이터 먼저 생성
python generate_poisoning_data.py --all
```

---

## 📚 더 자세한 가이드

- **전체 가이드**: `BACKDOOR_EXPERIMENT_README.md`
- **스크립트 사용법**: 각 Python 파일에서 `--help` 실행

```bash
python generate_poisoning_data.py --help
python finetune_backdoor.py --help
python evaluate_backdoor.py --help
python run_experiments.py --help
python visualize_results.py --help
```

---

## 🎉 시작하기

준비되셨나요? 이제 실행하세요!

```bash
# Windows
run_full_pipeline.bat

# Linux/Mac
python generate_poisoning_data.py --all && \
python run_experiments.py --quick && \
python visualize_results.py --summary results/quick_test_distilgpt2_200.json
```

**행운을 빕니다!** 🚀



