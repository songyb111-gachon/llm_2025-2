# GCG 공격 실습 빠른 시작 가이드

## 🚀 5분 안에 시작하기

### 1단계: 환경 설정 (1분)

```bash
# 방법 A: 자동 설정 (추천)
bash setup_experiment.sh

# 방법 B: 수동 설정
pip install -r requirements.txt

# GPU 확인 (선택사항)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 모델 미리 다운로드 (선택사항, 시간 절약)
python download_models.py --size medium
```

### 2단계: 빠른 테스트 (1분)

```bash
# 모델 로딩 및 기본 동작 확인
python quick_test.py gpt2
```

### 3단계: 본격 실험 시작

#### 옵션 A: 자동 스크립트 사용 (가장 쉬움) ⭐

```bash
# 기본 실행 (Pythia-1.4B, 10개 샘플)
bash run_experiment.sh

# 모델 변경
bash run_experiment.sh "gpt2" 10 250
```

#### 옵션 B: 기존 Suffix로 빠른 평가 (추천)

```bash
# Suffix 생성 없이 기존 Vicuna suffix로 테스트
# 100개 샘플을 빠르게 테스트 가능
bash run_with_existing_suffix.sh "EleutherAI/pythia-1.4b" 100
```

#### 옵션 C: 완전한 GCG 공격 (시간 많이 걸림)

```bash
# 새로운 suffix를 직접 생성
python simple_gcg_attack.py \
    --model_name "EleutherAI/pythia-1.4b" \
    --num_samples 10 \
    --num_steps 250
```

### 4단계: 결과 확인

```bash
# 결과 분석
python analyze_results.py results.json

# 또는 직접 확인
cat results.json | python -m json.tool
```

## 📋 추천 실험 순서

### 실험 1: 빠른 평가 (30분)
기존 Vicuna suffix로 여러 모델 비교

```bash
# GPT-2
bash run_with_existing_suffix.sh "gpt2" 100

# Pythia-1.4B
bash run_with_existing_suffix.sh "EleutherAI/pythia-1.4b" 100

# 결과 비교
python analyze_results.py results_existing_suffix_gpt2.json results_existing_suffix_EleutherAI_pythia-1.4b.json
```

### 실험 2: 심화 실험 (몇 시간)
특정 모델에 최적화된 suffix 생성

```bash
# 10개 샘플로 새 suffix 생성
bash run_experiment.sh "EleutherAI/pythia-1.4b" 10 250
```

## 🔧 문제 해결

### CUDA Out of Memory
```bash
# 더 작은 모델 사용
bash run_experiment.sh "gpt2" 5 200

# 또는 CPU 모드
python simple_gcg_attack.py --model_name "gpt2" --device cpu --num_samples 5
```

### 다운로드 실패
```bash
# 수동 다운로드
wget https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack_artifacts/GCG/white_box/vicuna-13b-v1.5.json
```

## 📊 예상 결과

| 모델 | 성공률 (기존 suffix) | 소요 시간 |
|------|---------------------|----------|
| GPT-2 | ~30-50% | 10-20분 (100개) |
| Pythia-1.4B | ~40-60% | 15-30분 (100개) |
| Pythia-2.8B | ~35-55% | 20-40분 (100개) |

*실제 결과는 환경에 따라 다를 수 있습니다*

## 💡 팁

1. **처음 시작**: `run_with_existing_suffix.sh`로 빠르게 테스트
2. **서버 실행**: `nohup`으로 백그라운드 실행
3. **메모리 부족**: 더 작은 모델(gpt2) 또는 샘플 수 줄이기
4. **시간 절약**: 기존 suffix 사용 (suffix 생성이 오래 걸림)

## 📝 과제 제출용 예시

```bash
# 1. Pythia-1.4B로 100개 테스트
bash run_with_existing_suffix.sh "EleutherAI/pythia-1.4b" 100

# 2. 결과 분석
python analyze_results.py results_existing_suffix_EleutherAI_pythia-1.4b.json

# 결과 파일: results_existing_suffix_EleutherAI_pythia-1.4b.json
# 이 파일에 성공률이 포함되어 있습니다
```

