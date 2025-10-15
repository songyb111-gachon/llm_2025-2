#!/bin/bash

# 기존 Vicuna suffix를 사용한 테스트 스크립트
# (Suffix 생성 없이 빠르게 테스트)

echo "=========================================="
echo "기존 Suffix로 빠른 테스트"
echo "=========================================="

MODEL_NAME=${1:-"EleutherAI/pythia-1.4b"}
NUM_SAMPLES=${2:-100}

echo ""
echo "모델: $MODEL_NAME"
echo "샘플 수: $NUM_SAMPLES"
echo ""

# Artifacts 다운로드 (없는 경우)
if [ ! -f "vicuna-13b-v1.5.json" ]; then
    echo "Artifacts 다운로드 중..."
    python download_artifacts.py
fi

# 기존 suffix로 테스트
python simple_gcg_attack.py \
    --model_name "$MODEL_NAME" \
    --num_samples $NUM_SAMPLES \
    --output_file "results_existing_suffix_$(echo $MODEL_NAME | tr '/' '_').json" \
    --use_existing_suffix \
    --device cuda

echo ""
echo "=========================================="
echo "테스트 완료!"
echo "=========================================="

