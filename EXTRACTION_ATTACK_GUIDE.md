# Model Extraction Attack Guide

## 개요

이 프로젝트는 작은 오픈 소스 LLM 모델에 대한 **Model Extraction Attack**을 구현합니다. Victim model (GPT-2)의 출력을 이용하여 더 작은 adversary model (DistilGPT-2)을 학습시켜 victim model을 복제하는 실험입니다.

## 실험 설정

### 모델
- **Victim Model**: GPT-2 (124M parameters)
- **Adversary Model**: DistilGPT-2 (82M parameters)
  - Baseline: Pretrained DistilGPT-2
  - Attack: Victim의 logits로 fine-tuned DistilGPT-2

### 데이터셋
- **Wikitext-2-raw-v1** (18.26MB)
- **Wikitext-103** (741.41MB)

### 평가 지표
1. **Perplexity**: 모델의 언어 모델링 성능 측정 (낮을수록 좋음)
2. **Accuracy**: 다음 토큰 예측 정확도 (높을수록 좋음)
3. **Fidelity@top-1**: Adversary 모델이 victim 모델의 예측을 얼마나 잘 따라하는지 측정 (높을수록 좋음)

## 설치

### 필수 패키지

```bash
pip install torch transformers datasets numpy tqdm matplotlib seaborn pandas
```

또는 requirements.txt에 추가:

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

## 사용 방법

### 1. 단일 실험 실행

기본 설정으로 실험 실행:

```bash
python extraction_attack.py \
    --victim_model gpt2 \
    --adversary_model distilgpt2 \
    --dataset wikitext-2-raw-v1 \
    --train_samples 1000 \
    --test_samples 500 \
    --batch_size 32 \
    --epochs 3 \
    --output_dir extraction_results \
    --save_model
```

### 2. 배치 실험 실행

Windows:
```bash
run_extraction.bat
```

Linux/Mac:
```bash
chmod +x run_extraction.sh
./run_extraction.sh
```

이 스크립트는 다음 실험을 자동으로 실행합니다:
- 1,000개 샘플로 학습
- 10,000개 샘플로 학습

### 3. 결과 비교 및 시각화

```bash
python compare_extraction_results.py \
    --results_dir extraction_results \
    --output_dir extraction_visualizations
```

이 스크립트는 다음을 생성합니다:
- 비교 테이블 (CSV)
- 메트릭 비교 그래프
- 학습 곡선 그래프
- 메트릭 히트맵

## 실험 파라미터

### 모델 설정
- `--victim_model`: Victim 모델 이름 (기본: gpt2)
- `--adversary_model`: Adversary 모델 이름 (기본: distilgpt2)

### 데이터셋 설정
- `--dataset`: 데이터셋 선택 (wikitext-2-raw-v1 또는 wikitext)
- `--train_samples`: 학습 샘플 수 (기본: 1000)
- `--test_samples`: 테스트 샘플 수 (기본: 500)
- `--max_length`: 최대 시퀀스 길이 (기본: 128)

### 학습 설정
- `--batch_size`: 배치 크기 (기본: 32)
- `--epochs`: 학습 에포크 수 (기본: 3)
- `--learning_rate`: 학습률 (기본: 5e-5)
- `--temperature`: 증류 온도 (기본: 1.0)
- `--alpha`: KL divergence와 CE loss의 균형 (기본: 0.5)

### 출력 설정
- `--output_dir`: 결과 저장 디렉토리 (기본: extraction_results)
- `--save_model`: Fine-tuned 모델 저장 여부

## 예상 결과

### 과제 예시와 동일한 설정

**Dataset**: wikitext-2-raw-v1  
**Test samples**: 500개  
**Training samples**: 1,000 / 10,000  
**Batch size**: 32  
**Using logits**: True

| Model | Perplexity | Accuracy | Fidelity@top-1 |
|-------|-----------|----------|----------------|
| gpt-2 (victim) | 51.32 | 32.53% | 100% |
| distilgpt2 (baseline) | 51.32 | 28.10% | 79.40% |
| finetuned distilgpt2 (with 1000 data) | 84.68 | 27.46% | 99.00% |
| finetuned distilgpt2 (with 10000 data) | 173.18 | 21.54% | 98.80% |

## 작동 원리

### 1. Extraction Phase
Victim model (GPT-2)에 학습 데이터를 입력하여 logits을 추출합니다.

```python
with torch.no_grad():
    outputs = victim_model(input_ids=input_ids)
    victim_logits = outputs.logits
```

### 2. Fine-tuning Phase
Adversary model (DistilGPT-2)을 victim의 logits로 학습시킵니다.

**Loss Function**:
```
Total Loss = α × KL_Divergence(Student || Teacher) + (1-α) × CrossEntropy(Student, Labels)
```

- **KL Divergence**: Adversary가 victim의 출력 분포를 따라하도록 학습
- **Cross Entropy**: 원래 언어 모델링 목표 유지

### 3. Evaluation Phase
세 가지 모델을 비교 평가합니다:
1. Victim (GPT-2) - 기준점
2. Baseline (Pretrained DistilGPT-2) - 공격 전
3. Fine-tuned (Attacked DistilGPT-2) - 공격 후

## 디렉토리 구조

```
.
├── extraction_attack.py              # 메인 실험 스크립트
├── run_extraction.bat                # Windows 배치 실험
├── run_extraction.sh                 # Linux/Mac 배치 실험
├── compare_extraction_results.py     # 결과 비교 및 시각화
├── EXTRACTION_ATTACK_GUIDE.md        # 이 가이드 문서
├── extraction_results/               # 실험 결과
│   ├── results_1000samples.json
│   ├── results_10000samples.json
│   ├── finetuned_distilgpt2_1000/   # 저장된 모델
│   └── finetuned_distilgpt2_10000/
└── extraction_visualizations/        # 시각화 결과
    ├── comparison_table.csv
    ├── extraction_comparison.png
    ├── training_curves.png
    └── metrics_heatmap.png
```

## 고급 사용법

### 다른 모델로 실험

```bash
# GPT-2 Medium vs DistilGPT-2
python extraction_attack.py \
    --victim_model gpt2-medium \
    --adversary_model distilgpt2 \
    --train_samples 5000

# GPT-2 vs GPT-2 Small (자기 복제)
python extraction_attack.py \
    --victim_model gpt2 \
    --adversary_model gpt2 \
    --train_samples 1000
```

### Loss 함수 튜닝

Alpha 값을 조정하여 fidelity와 accuracy의 균형을 맞출 수 있습니다:

```bash
# Fidelity 우선 (KL divergence 강조)
python extraction_attack.py --alpha 0.8

# Accuracy 우선 (Cross entropy 강조)
python extraction_attack.py --alpha 0.2
```

### Temperature Scaling

Temperature를 조정하여 knowledge distillation의 강도를 조절할 수 있습니다:

```bash
# 더 부드러운 확률 분포 (더 많은 정보 전달)
python extraction_attack.py --temperature 2.0

# 더 날카로운 확률 분포
python extraction_attack.py --temperature 0.5
```

## 결과 해석

### Perplexity
- Victim보다 낮으면: Adversary가 더 나은 언어 모델
- Victim보다 높으면: Adversary가 덜 정확한 언어 모델
- Fine-tuning 후 증가할 수 있음: Victim의 특정 예측 패턴을 따라하느라 일반화 성능 저하

### Accuracy
- 다음 토큰을 정확히 맞추는 비율
- 언어 모델의 실제 성능 지표

### Fidelity
- Adversary가 victim의 예측을 얼마나 잘 복제하는지 측정
- 100%에 가까울수록 성공적인 extraction
- 높은 fidelity = victim 모델의 지식 성공적 복제

### Trade-offs
- **Fidelity vs Accuracy**: Victim의 예측을 따라하면 fidelity는 높아지지만, victim의 오류도 함께 학습될 수 있음
- **Data Size**: 더 많은 데이터로 학습하면 fidelity는 높아지지만, overfitting으로 perplexity가 높아질 수 있음

## 문제 해결

### CUDA Out of Memory
배치 크기를 줄이거나 시퀀스 길이를 줄입니다:
```bash
python extraction_attack.py --batch_size 16 --max_length 64
```

### 학습이 너무 느림
- GPU가 있다면 CUDA 설치 확인
- 샘플 수를 줄여서 테스트
- 더 작은 모델 사용

### Fidelity가 너무 낮음
- Alpha 값을 높여서 KL divergence 강조
- Temperature를 높여서 더 많은 정보 전달
- 학습 에포크 수 증가

## 참고 자료

### Model Extraction Attacks
- [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)
- [High Accuracy and High Fidelity Extraction of Neural Networks](https://arxiv.org/abs/1909.01838)

### Knowledge Distillation
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

## 라이선스

이 프로젝트는 교육 목적으로 제공됩니다.

## 기여

버그 리포트, 기능 제안 등은 이슈로 등록해주세요.

