# Model Extraction Attack - 빠른 시작 가이드

## 5분 안에 시작하기

### 1단계: 환경 설정

```bash
# 필수 패키지 설치
pip install torch transformers datasets numpy tqdm matplotlib seaborn pandas
```

### 2단계: 빠른 테스트 (선택사항)

설정이 올바른지 확인하기 위한 빠른 테스트 (1-2분 소요):

**Windows:**
```bash
quick_extraction_test.bat
```

**Linux/Mac:**
```bash
python quick_extraction_test.py
```

### 3단계: 전체 실험 실행

**Windows:**
```bash
run_extraction.bat
```

**Linux/Mac:**
```bash
chmod +x run_extraction.sh
./run_extraction.sh
```

### 4단계: 결과 확인

```bash
python compare_extraction_results.py
```

## 무엇을 하는 프로젝트인가?

이 프로젝트는 **Model Extraction Attack**을 구현합니다:

1. **Victim Model** (GPT-2): 공격 대상 모델
2. **Adversary Model** (DistilGPT-2): 공격자가 학습시키는 모델

### 공격 과정:
```
1. Victim에게 질의 → logits 추출
2. Adversary를 victim의 logits로 학습
3. Adversary가 victim을 모방하게 됨
```

### 평가 지표:
- **Perplexity**: 언어 모델 성능 (낮을수록 좋음)
- **Accuracy**: 다음 토큰 예측 정확도 (높을수록 좋음)
- **Fidelity**: Victim 모델 모방 정도 (높을수록 성공적인 공격)

## 예상 실행 시간

- 빠른 테스트: 1-2분
- 1,000 샘플 실험: 10-15분
- 10,000 샘플 실험: 30-60분

(GPU 사용 시 더 빠름)

## 예상 결과

| Model | Perplexity | Accuracy | Fidelity@top-1 |
|-------|-----------|----------|----------------|
| GPT-2 (victim) | ~51 | ~32% | 100% |
| DistilGPT-2 (baseline) | ~51 | ~28% | ~79% |
| Fine-tuned (1000 samples) | ~85 | ~27% | ~99% |
| Fine-tuned (10000 samples) | ~173 | ~22% | ~99% |

## 주요 파일

| 파일 | 설명 |
|------|------|
| `extraction_attack.py` | 메인 실험 스크립트 |
| `run_extraction.bat` | Windows 배치 실험 |
| `run_extraction.sh` | Linux/Mac 배치 실험 |
| `compare_extraction_results.py` | 결과 비교 및 시각화 |
| `quick_extraction_test.py` | 빠른 테스트 |
| `EXTRACTION_ATTACK_GUIDE.md` | 상세 가이드 |

## 커스터마이징

### 다른 데이터 크기로 실험

```bash
python extraction_attack.py --train_samples 5000 --test_samples 500
```

### 다른 모델로 실험

```bash
python extraction_attack.py --victim_model gpt2-medium --adversary_model distilgpt2
```

### Loss 함수 조정

```bash
# Fidelity 우선
python extraction_attack.py --alpha 0.8

# Accuracy 우선
python extraction_attack.py --alpha 0.2
```

## 문제 해결

### 메모리 부족
```bash
python extraction_attack.py --batch_size 8 --max_length 64
```

### 모델 다운로드 실패
- 인터넷 연결 확인
- Hugging Face Hub 접근 가능 여부 확인

### 느린 실행 속도
- CUDA 설치 확인 (GPU 사용)
- 샘플 수 줄이기
- 배치 크기 늘리기 (메모리가 충분하다면)

## 다음 단계

1. ✅ 빠른 테스트로 설정 확인
2. ✅ 전체 실험 실행
3. ✅ 결과 시각화 확인
4. 📊 다양한 파라미터로 실험
5. 📝 결과 분석 및 리포트 작성

## 도움말

자세한 내용은 `EXTRACTION_ATTACK_GUIDE.md`를 참고하세요.

질문이나 문제가 있다면 이슈로 등록해주세요.


