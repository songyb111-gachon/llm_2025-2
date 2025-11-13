# Model Extraction Attack - 개선 사항

## 🔧 수정 내역

### 1. Perplexity 계산 개선

**문제점:**
- 기존 코드는 `outputs.loss`를 사용했는데, 이미 평균된 loss에 다시 토큰 수를 곱해서 부정확한 계산
- 결과: 매우 높은 perplexity (731, 1001 등)

**해결책:**
```python
# 토큰 단위로 loss를 계산하고 valid token에만 적용
criterion = nn.CrossEntropyLoss(reduction='none')
loss_per_token = criterion(shift_logits, shift_labels)
valid_loss = (loss_per_token * shift_mask).sum()
valid_tokens = shift_mask.sum()
avg_loss = valid_loss / valid_tokens
perplexity = np.exp(avg_loss)
```

### 2. KL Divergence Loss 개선

**문제점:**
- `reduction='batchmean'`이 마스크를 고려하지 않음
- Padding token에 대해서도 loss를 계산

**해결책:**
```python
# Valid token에 대해서만 KL divergence 계산
kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
kl_loss = (kl_per_token * flat_mask).sum() / flat_mask.sum()
```

### 3. 학습 파라미터 최적화

**변경 사항:**
| 파라미터 | 이전 | 개선 | 이유 |
|---------|------|------|------|
| `epochs` | 3 | 5 | 더 많은 학습 필요 |
| `temperature` | 1.0 | 2.0 | 더 부드러운 확률 분포 (더 많은 정보) |
| `alpha` | 0.5 | 0.7 | KL divergence에 더 가중치 (fidelity 향상) |

**Temperature 효과:**
- `T = 1.0`: 원래 확률 분포 사용
- `T = 2.0`: 더 부드러운 분포 → Teacher의 "dark knowledge" 더 많이 전달

**Alpha 효과:**
- `α = 0.5`: KL과 CE를 동등하게
- `α = 0.7`: KL에 70%, CE에 30% 가중치 → Fidelity 우선

## 📊 예상 개선 효과

### 이전 결과 (잘못된 계산)
| Model | Perplexity | Accuracy | Fidelity@top-1 |
|-------|-----------|----------|----------------|
| GPT-2 (victim) | 731.68 ⚠️ | 29.96% | 100% |
| DistilGPT-2 (baseline) | 1001.18 ⚠️ | 25.45% | 62.96% |
| Fine-tuned (1000) | 867.31 ⚠️ | 27.42% | 66.77% ⚠️ |

### 개선 후 예상 결과
| Model | Perplexity | Accuracy | Fidelity@top-1 |
|-------|-----------|----------|----------------|
| GPT-2 (victim) | ~50-60 ✅ | ~30-33% | 100% |
| DistilGPT-2 (baseline) | ~50-60 ✅ | ~26-28% | ~75-80% |
| Fine-tuned (1000) | ~80-90 ✅ | ~27-28% | **~95-99%** ✅ |

## 🚀 재실행 방법

### 1. 이전 결과 삭제 (선택사항)
```bash
rm -rf extraction_results/
rm -rf extraction_visualizations/
```

### 2. 개선된 코드로 재실행
```bash
# Linux/Mac
./run_extraction.sh

# Windows
run_extraction.bat
```

### 3. 결과 확인
```bash
python compare_extraction_results.py
```

## 🔍 개선 사항 상세 설명

### Why Temperature = 2.0?

Knowledge Distillation에서 temperature는 확률 분포를 부드럽게 만듭니다:

```python
# T=1.0: [0.9, 0.05, 0.03, 0.02] (날카로운 분포)
# T=2.0: [0.6, 0.2, 0.15, 0.05] (부드러운 분포)
```

부드러운 분포는 모델이 "왜" 그 예측을 했는지에 대한 정보를 더 많이 포함합니다.
- Top-1 답뿐만 아니라 다른 가능한 답들의 상대적 확률도 학습
- Victim의 "dark knowledge" 전달

### Why Alpha = 0.7?

Loss = 0.7 × KL + 0.3 × CE

- **KL Divergence (70%)**: Victim의 예측 분포를 따라하도록 (Fidelity ↑)
- **Cross Entropy (30%)**: 정답 레이블도 맞추도록 (Accuracy 유지)

Alpha 조정 예시:
```bash
# Fidelity 최대화
python extraction_attack.py --alpha 0.9

# Accuracy 최대화
python extraction_attack.py --alpha 0.3
```

### Why Epochs = 5?

더 많은 에포크로 학습:
- Victim의 패턴을 더 깊이 학습
- Fidelity 향상
- 단, 너무 많으면 과적합 위험 (Epoch 7-10 이상은 주의)

## 📈 성능 지표 해석

### Perplexity
- **낮을수록 좋음**: 모델의 불확실성이 낮음
- **50-100**: 좋은 언어 모델
- **100-200**: 괜찮은 성능
- **200+**: 과적합 또는 문제 있음

### Fidelity@top-1
- **95%+**: 매우 성공적인 extraction
- **85-95%**: 성공적인 extraction
- **70-85%**: 부분적 성공
- **<70%**: 개선 필요

## 🎯 추가 최적화 팁

### 1. 더 많은 데이터
```bash
python extraction_attack.py --train_samples 20000
```

### 2. 더 큰 배치 크기 (GPU 메모리가 충분하다면)
```bash
python extraction_attack.py --batch_size 64
```

### 3. 더 높은 Temperature
```bash
python extraction_attack.py --temperature 3.0
```

### 4. 학습률 조정
```bash
python extraction_attack.py --learning_rate 1e-4
```

## 🐛 문제 해결

### Perplexity가 여전히 너무 높다면
1. Temperature를 더 높이기 (3.0-4.0)
2. Alpha를 더 높이기 (0.8-0.9)
3. 학습률을 높이기 (1e-4)

### Fidelity가 개선되지 않는다면
1. Epochs 늘리기 (7-10)
2. Alpha를 0.8-0.9로 높이기
3. Temperature를 3.0-4.0으로 높이기

### 과적합이 발생한다면
- Perplexity가 너무 높고 Accuracy가 낮다면
- Epochs 줄이기 (3-4)
- Alpha를 낮추기 (0.5-0.6)

## 📚 참고 자료

### Knowledge Distillation
- Hinton et al. (2015): Temperature = 2-20 추천
- 본 실험: Temperature = 2.0으로 시작

### Model Extraction
- Fidelity 95%+ 달성 시 성공적인 extraction
- Trade-off: Fidelity ↑ vs Generalization ↓

## 🔄 변경 이력

**2025-11-13**
- Perplexity 계산 수정 (토큰 단위 정확한 계산)
- KL divergence 마스킹 적용
- 학습 파라미터 최적화 (T=2.0, α=0.7, epochs=5)

