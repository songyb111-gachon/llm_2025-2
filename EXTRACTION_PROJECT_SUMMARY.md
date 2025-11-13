# Model Extraction Attack 프로젝트 요약

## 프로젝트 개요

이 프로젝트는 소규모 오픈소스 LLM(Language Model)에 대한 **Model Extraction Attack**을 구현하고 평가합니다.

### 핵심 개념

**Model Extraction Attack**은 블랙박스 모델의 입출력만을 관찰하여 해당 모델을 복제하는 공격 기법입니다.

```
┌─────────────┐        Query         ┌──────────────┐
│  Adversary  │ ──────────────────> │    Victim    │
│   Model     │                      │    Model     │
│ (DistilGPT2)│ <────────────────── │    (GPT-2)   │
└─────────────┘      Logits          └──────────────┘
       │                                     
       │ Learn from                          
       │ Victim's outputs                    
       ▼                                     
┌─────────────┐                             
│  Cloned     │                             
│   Model     │                             
└─────────────┘                             
```

## 실험 설정

| 구성 요소 | 설명 |
|----------|------|
| **Victim** | GPT-2 (124M parameters) |
| **Adversary** | DistilGPT-2 (82M parameters) |
| **Dataset** | Wikitext-2-raw-v1 (18.26MB) |
| **Training Size** | 1,000 / 10,000 samples |
| **Test Size** | 500 samples |
| **Batch Size** | 32 |
| **Method** | Knowledge Distillation using Logits |

## 평가 지표

### 1. Perplexity (PPL)
```
PPL = exp(average cross-entropy loss)
```
- 언어 모델의 불확실성 측정
- **낮을수록 좋음**
- 모델이 다음 단어를 얼마나 잘 예측하는지

### 2. Accuracy (ACC)
```
ACC = (correct predictions) / (total predictions) × 100%
```
- 다음 토큰 예측의 정확도
- **높을수록 좋음**
- 실제 성능 지표

### 3. Fidelity@top-1 (FID)
```
FID = (matching predictions) / (total predictions) × 100%
```
- Adversary가 Victim의 예측을 얼마나 따라하는지
- **높을수록 성공적인 공격**
- 모델 복제 정도 측정

## 기대 결과

### 실험 1: 1,000 샘플 학습

| Model | Perplexity | Accuracy | Fidelity@top-1 |
|-------|-----------|----------|----------------|
| GPT-2 (victim) | 51.32 | 32.53% | 100.00% |
| DistilGPT-2 (baseline) | 51.32 | 28.10% | 79.40% |
| **Fine-tuned (1000)** | **84.68** | **27.46%** | **99.00%** |

**분석:**
- ✅ Fidelity가 79% → 99%로 크게 증가
- ⚠️ Perplexity가 51 → 85로 증가 (일반화 성능 저하)
- ✅ Victim 모델의 행동을 성공적으로 복제

### 실험 2: 10,000 샘플 학습

| Model | Perplexity | Accuracy | Fidelity@top-1 |
|-------|-----------|----------|----------------|
| GPT-2 (victim) | 51.32 | 32.53% | 100.00% |
| DistilGPT-2 (baseline) | 51.32 | 28.10% | 79.40% |
| **Fine-tuned (10000)** | **173.18** | **21.54%** | **98.80%** |

**분석:**
- ✅ 높은 Fidelity 유지 (98.80%)
- ⚠️ Perplexity 더욱 증가 (173.18)
- ⚠️ Accuracy 감소 (21.54%)
- 📊 **과적합 현상**: Victim의 특정 패턴에 과도하게 학습

## 핵심 발견

### Trade-offs

1. **Fidelity vs Generalization**
   ```
   ↑ More Training Data → ↑ Fidelity
                         → ↓ Generalization (↑ Perplexity)
   ```

2. **Copying vs Understanding**
   - Adversary는 Victim을 "이해"하기보다는 "복사"함
   - Victim의 오류까지 함께 학습

3. **Attack Success**
   - Fidelity 99%: 매우 성공적인 extraction
   - 단, 실용적 성능은 저하

## 방법론

### Knowledge Distillation Loss

```python
Total_Loss = α × KL_Divergence(Student || Teacher) + (1-α) × CrossEntropy(Student, Labels)
```

**구성 요소:**
- `α`: KL과 CE 사이의 균형 (기본값: 0.5)
- `KL_Divergence`: Victim의 출력 분포를 따라하도록
- `CrossEntropy`: 원래 언어 모델 목표 유지

### Temperature Scaling

```python
Softmax(logits / T)
```

- `T = 1`: 원래 확률 분포
- `T > 1`: 더 부드러운 분포 (더 많은 정보)
- `T < 1`: 더 날카로운 분포

## 프로젝트 구조

```
extraction-attack/
│
├── extraction_attack.py              # 메인 실험 코드
├── compare_extraction_results.py     # 결과 분석
├── quick_extraction_test.py          # 빠른 테스트
│
├── run_extraction.bat                # Windows 실행
├── run_extraction.sh                 # Linux/Mac 실행
├── quick_extraction_test.bat         # Windows 빠른 테스트
│
├── EXTRACTION_QUICKSTART.md          # 빠른 시작 가이드
├── EXTRACTION_ATTACK_GUIDE.md        # 상세 가이드
└── EXTRACTION_PROJECT_SUMMARY.md     # 이 문서
```

## 실행 방법

### 빠른 시작 (1-2분)
```bash
# Windows
quick_extraction_test.bat

# Linux/Mac
python quick_extraction_test.py
```

### 전체 실험 (30-60분)
```bash
# Windows
run_extraction.bat

# Linux/Mac
./run_extraction.sh
```

### 결과 시각화
```bash
python compare_extraction_results.py
```

## 실험 파라미터 조정

### Alpha 값 조정
```bash
# Fidelity 최적화
python extraction_attack.py --alpha 0.8

# Accuracy 최적화
python extraction_attack.py --alpha 0.2
```

### Temperature 조정
```bash
# 부드러운 확률 분포
python extraction_attack.py --temperature 2.0

# 날카로운 확률 분포
python extraction_attack.py --temperature 0.5
```

### 데이터 크기 조정
```bash
python extraction_attack.py --train_samples 5000
```

## 보안 및 방어 대책

### 공격이 성공하는 이유
1. Logits 정보 노출
2. 충분한 query 가능
3. 유사한 모델 아키텍처

### 방어 방법
1. **Logits 은닉**: Top-k만 반환
2. **Query 제한**: 요청 수 제한
3. **Noise 추가**: 출력에 노이즈 주입
4. **Watermarking**: 모델 워터마킹

## 윤리적 고려사항

⚠️ **주의**: 이 프로젝트는 교육 목적입니다.

- 실제 상용 모델에 대한 무단 공격은 불법입니다
- 모델 제공자의 서비스 약관을 준수해야 합니다
- 연구 및 학습 목적으로만 사용하세요

## 학습 목표

이 프로젝트를 통해 다음을 학습할 수 있습니다:

1. ✅ Model Extraction Attack 원리
2. ✅ Knowledge Distillation 기법
3. ✅ 언어 모델 평가 지표
4. ✅ PyTorch와 Transformers 라이브러리 사용
5. ✅ 실험 설계 및 결과 분석

## 확장 아이디어

### 1. 다양한 모델 조합
```
GPT-2 → GPT-2-Medium
GPT-2 → GPT-Neo
BERT → DistilBERT
```

### 2. 방어 기법 구현
- Output perturbation
- Query budget
- Confidence masking

### 3. 다른 태스크에 적용
- Text classification
- Question answering
- Summarization

## 참고 문헌

### 핵심 논문
1. **Model Extraction**
   - Tramèr et al. (2016) - "Stealing Machine Learning Models via Prediction APIs"
   - Jagielski et al. (2020) - "High Accuracy and High Fidelity Extraction of Neural Networks"

2. **Knowledge Distillation**
   - Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
   - Sanh et al. (2019) - "DistilBERT"

3. **Defense Mechanisms**
   - Orekondy et al. (2019) - "Prediction Poisoning"
   - Kariyappa et al. (2021) - "Defending Against Model Extraction"

## 지원 및 문의

- 📚 상세 가이드: `EXTRACTION_ATTACK_GUIDE.md`
- 🚀 빠른 시작: `EXTRACTION_QUICKSTART.md`
- 💻 GitHub Issues: 버그 리포트 및 기능 제안

## 라이선스

이 프로젝트는 교육 목적으로 제공됩니다.


