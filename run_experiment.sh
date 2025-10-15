#!/bin/bash

# GCG 공격 실험 실행 스크립트
# 서버에서 실행하기 위한 셸 스크립트

echo "=========================================="
echo "GCG Attack 실험 시작"
echo "=========================================="

# 1. Attack artifacts 다운로드
echo ""
echo "[1/4] Attack artifacts 다운로드 중..."
python download_artifacts.py

if [ ! -f "vicuna-13b-v1.5.json" ]; then
    echo "Error: artifacts 다운로드 실패"
    exit 1
fi

# 2. 모델 선택 (인자로 받거나 기본값 사용)
MODEL_NAME=${1:-"EleutherAI/pythia-1.4b"}
NUM_SAMPLES=${2:-10}
NUM_STEPS=${3:-250}

echo ""
echo "[2/4] 실험 설정"
echo "  - 모델: $MODEL_NAME"
echo "  - 샘플 수: $NUM_SAMPLES"
echo "  - 최적화 스텝: $NUM_STEPS"

# 3. GCG 공격 실행
echo ""
echo "[3/4] GCG 공격 실행 중..."
python simple_gcg_attack.py \
    --model_name "$MODEL_NAME" \
    --num_samples $NUM_SAMPLES \
    --num_steps $NUM_STEPS \
    --suffix_length 20 \
    --output_file "results_$(echo $MODEL_NAME | tr '/' '_').json" \
    --device cuda

# 4. 결과 출력
echo ""
echo "[4/4] 실험 완료!"
echo "결과 파일: results_$(echo $MODEL_NAME | tr '/' '_').json"
echo ""
echo "=========================================="

