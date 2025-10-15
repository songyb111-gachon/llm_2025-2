# GCG Attack 실습

이 프로젝트는 GCG (Greedy Coordinate Gradient) 공격을 구현하고 테스트하는 실습 코드입니다.

## 환경 설정

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. GPU 확인 (선택사항)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 실행 방법

### 방법 1: 자동 실행 스크립트 사용 (추천)

```bash
# 기본 실행 (Pythia-1.4B, 10개 샘플)
bash run_experiment.sh

# 모델 및 샘플 수 지정
bash run_experiment.sh "EleutherAI/pythia-1.4b" 20 250

# 다른 모델 예시
bash run_experiment.sh "gpt2" 10 200
bash run_experiment.sh "tiiuae/falcon-7b" 5 300
```

### 방법 2: Python 직접 실행

#### Step 1: Artifacts 다운로드

```bash
python download_artifacts.py
```

#### Step 2: GCG 공격 실행

```bash
# 새로운 suffix 생성하여 공격
python simple_gcg_attack.py \
    --model_name "EleutherAI/pythia-1.4b" \
    --num_samples 10 \
    --num_steps 250 \
    --suffix_length 20 \
    --output_file results.json \
    --device cuda
```

#### Step 3 (선택): 기존 Vicuna suffix로 빠른 테스트

```bash
# Suffix 생성 없이 기존 suffix로 테스트
python simple_gcg_attack.py \
    --model_name "EleutherAI/pythia-1.4b" \
    --num_samples 100 \
    --use_existing_suffix \
    --output_file results_existing.json \
    --device cuda

# 또는 셸 스크립트 사용
bash run_with_existing_suffix.sh "EleutherAI/pythia-1.4b" 100
```

## 지원 모델

다음 모델들을 테스트할 수 있습니다:

- `EleutherAI/pythia-1.4b` - 1.4B 파라미터 (추천: 빠른 실험)
- `EleutherAI/pythia-2.8b` - 2.8B 파라미터
- `gpt2` - GPT-2 (124M)
- `gpt2-medium` - GPT-2 Medium (355M)
- `gpt2-large` - GPT-2 Large (774M)
- `tiiuae/falcon-7b` - Falcon 7B (GPU 메모리 많이 필요)
- `mosaicml/mpt-7b` - MPT 7B
- `timdettmers/guanaco-7b` - Guanaco 7B

## 실행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model_name` | HuggingFace 모델 이름 | (필수) |
| `--artifacts_path` | Artifacts JSON 경로 | `vicuna-13b-v1.5.json` |
| `--num_samples` | 테스트할 샘플 수 | 10 |
| `--num_steps` | GCG 최적화 반복 횟수 | 250 |
| `--suffix_length` | Adversarial suffix 토큰 길이 | 20 |
| `--output_file` | 결과 저장 파일 | `results.json` |
| `--use_existing_suffix` | 기존 suffix 재사용 (생성 건너뛰기) | False |
| `--device` | 디바이스 (cuda/cpu) | cuda |

## 결과 확인

실행이 완료되면 JSON 파일이 생성됩니다:

```json
{
  "model_name": "EleutherAI/pythia-1.4b",
  "num_samples": 10,
  "success_count": 7,
  "success_rate": 70.0,
  "results": [
    {
      "prompt": "Write a tutorial on how to...",
      "suffix": "generated adversarial suffix",
      "response": "model's response",
      "is_success": true
    }
  ]
}
```

## 서버 실행 예시

### GPU 서버에서 백그라운드 실행

```bash
# nohup으로 백그라운드 실행
nohup bash run_experiment.sh "EleutherAI/pythia-1.4b" 50 300 > experiment.log 2>&1 &

# 로그 확인
tail -f experiment.log
```

### 여러 모델 순차 실행

```bash
# 여러 모델 테스트 스크립트
for model in "EleutherAI/pythia-1.4b" "gpt2" "gpt2-medium"
do
    echo "Testing $model..."
    bash run_experiment.sh "$model" 20 250
    sleep 10
done
```

### SLURM 환경에서 실행

```bash
#!/bin/bash
#SBATCH --job-name=gcg_attack
#SBATCH --output=gcg_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module load python/3.10
module load cuda/11.8

source venv/bin/activate

bash run_experiment.sh "EleutherAI/pythia-1.4b" 100 500
```

## 주의사항

1. **메모리**: 큰 모델(7B 이상)은 GPU 메모리를 많이 사용합니다 (최소 24GB 권장)
2. **시간**: Suffix 생성은 오래 걸릴 수 있습니다 (샘플당 5-30분)
3. **윤리**: 이 코드는 교육 목적입니다. 실제 악의적 사용은 금지됩니다

## 문제 해결

### CUDA Out of Memory

```bash
# 더 작은 모델 사용
bash run_experiment.sh "gpt2" 10 200

# 또는 CPU 사용
python simple_gcg_attack.py --model_name "gpt2" --device cpu
```

### 다운로드 실패

```bash
# 수동 다운로드
wget https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack_artifacts/GCG/white_box/vicuna-13b-v1.5.json
```

## 참고자료

- [GCG Paper](https://arxiv.org/abs/2307.15043)
- [JailbreakBench](https://jailbreakbench.github.io/)
- [Attack Artifacts](https://github.com/JailbreakBench/artifacts)

