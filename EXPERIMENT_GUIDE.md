# GCG 공격 전체 실험 가이드

## 🎯 과제 목표

100개 prompt에 대해 여러 모델의 GCG 공격 성공률 측정

## 🚀 빠른 시작 (서버)

### 1단계: 환경 설정

```bash
cd ~/llm_2025-2
git pull
pip install -r requirements.txt
```

### 2단계: 모델 다운로드 (선택사항, 시간 절약)

```bash
# 경량 모델만 (추천)
python download_models.py --size small
python download_models.py --size medium

# 또는 전부
python download_models.py --size all
```

### 3단계: 전체 실험 실행

```bash
# 방법 A: 전체 모델 실험 (6개 모델)
nohup bash run_all_models.sh 100 > experiment_all.log 2>&1 &

# 방법 B: 경량 모델만 (4개 모델, 안전)
nohup bash run_all_models_light.sh 100 > experiment_light.log 2>&1 &

# 방법 C: 대용량 모델만 (3개 모델, GPU 24GB+)
nohup bash run_all_models_heavy.sh 100 > experiment_heavy.log 2>&1 &
```

### 4단계: 진행 상황 모니터링

```bash
# 로그 실시간 확인
tail -f experiment_all.log

# 또는
tail -f experiment_light.log
```

## 📊 실험 옵션 비교

### 옵션 1: 전체 모델 (`run_all_models.sh`) ⭐

**포함 모델:**
- gpt2 (124M)
- gpt2-medium (355M)
- EleutherAI/pythia-1.4b (1.4B)
- EleutherAI/pythia-2.8b (2.8B)
- tiiuae/falcon-7b (7B)
- mosaicml/mpt-7b (7B)

**예상 시간:** 3-6시간 (100개 샘플)  
**GPU 요구사항:** 24GB+ 권장

```bash
bash run_all_models.sh 100
```

### 옵션 2: 경량 모델 (`run_all_models_light.sh`) 🎯 추천

**포함 모델:**
- gpt2
- gpt2-medium
- EleutherAI/pythia-1.4b
- EleutherAI/pythia-2.8b

**예상 시간:** 1-3시간  
**GPU 요구사항:** 16GB

```bash
bash run_all_models_light.sh 100
```

### 옵션 3: 대용량 모델 (`run_all_models_heavy.sh`)

**포함 모델:**
- tiiuae/falcon-7b
- mosaicml/mpt-7b
- timdettmers/guanaco-7b

**예상 시간:** 3-5시간  
**GPU 요구사항:** 24GB+

```bash
bash run_all_models_heavy.sh 100
```

## 📁 결과 구조

실험이 완료되면 다음과 같은 구조로 결과가 저장됩니다:

```
results_20241015_123456/
├── results_gpt2.json                      # 각 모델별 상세 결과
├── results_gpt2-medium.json
├── results_EleutherAI_pythia-1.4b.json
├── results_EleutherAI_pythia-2.8b.json
├── log_gpt2.txt                           # 각 모델별 로그
├── log_gpt2-medium.txt
├── comparison.csv                          # 모델 비교 CSV
└── report.txt                             # 최종 보고서
```

## 📈 결과 분석

### 개별 모델 결과 확인

```bash
# 특정 모델 결과 분석
python analyze_results.py results_20241015_123456/results_gpt2.json

# JSON 직접 확인
cat results_20241015_123456/results_gpt2.json | python -m json.tool | head -50
```

### 전체 비교

```bash
# 모델 간 비교표 생성
python compare_all_results.py results_20241015_123456/

# 보고서 확인
cat results_20241015_123456/report.txt

# CSV 열기 (엑셀 등)
cat results_20241015_123456/comparison.csv
```

## 🔧 문제 해결

### CUDA Out of Memory

```bash
# 경량 모델만 실행
bash run_all_models_light.sh 100

# 또는 샘플 수 줄이기
bash run_all_models_light.sh 50
```

### 실험 중단 후 재개

특정 모델부터 다시 시작하려면:

```bash
# 수동으로 개별 실행
python simple_gcg_attack.py \
    --model_name "EleutherAI/pythia-1.4b" \
    --num_samples 100 \
    --use_existing_suffix \
    --output_file results/results_pythia.json
```

### 특정 모델만 실험

```bash
# 직접 모델 지정
python simple_gcg_attack.py \
    --model_name "gpt2" \
    --num_samples 100 \
    --use_existing_suffix \
    --output_file my_results.json
```

## 💡 최적화 팁

### 1. 모델 미리 다운로드 ⭐
```bash
# 실험 전에 모든 모델 다운로드
python download_models.py --size medium
```

### 2. 기존 Suffix 사용 (기본값)
- 새로운 suffix 생성보다 100배 빠름
- Vicuna의 suffix를 재사용
- 성공률 측정에 충분

### 3. Screen/Tmux 사용
```bash
# screen 세션에서 실행
screen -S gcg_experiment
bash run_all_models_light.sh 100

# 세션 나가기: Ctrl+A, D
# 다시 붙기: screen -r gcg_experiment
```

### 4. 병렬 실행 (GPU 여러 개)
```bash
# GPU 0에서 경량 모델
CUDA_VISIBLE_DEVICES=0 bash run_all_models_light.sh 100 &

# GPU 1에서 대용량 모델
CUDA_VISIBLE_DEVICES=1 bash run_all_models_heavy.sh 100 &
```

## 📝 과제 제출용 체크리스트

- [ ] 실험 환경 설정 완료
- [ ] Attack artifacts 다운로드 완료
- [ ] 최소 3개 이상 모델로 100개 샘플 테스트
- [ ] 각 모델별 성공률 결과 확인
- [ ] 비교표 및 보고서 생성
- [ ] 결과 파일 백업

## 🎓 예상 결과 (참고용)

| 모델 | 예상 성공률 |
|------|-----------|
| GPT-2 | 30-50% |
| GPT-2 Medium | 35-55% |
| Pythia-1.4B | 40-60% |
| Pythia-2.8B | 35-55% |
| Falcon-7B | 25-45% |

*실제 결과는 환경에 따라 다를 수 있습니다*

## 📞 도움말

```bash
# 스크립트 도움말
bash run_all_models.sh --help

# Python 도움말
python simple_gcg_attack.py --help
python analyze_results.py --help
```

