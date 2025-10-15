#!/bin/bash

# 대용량 모델 실험 (GPU 24GB+ 필요)
# Falcon-7B, MPT-7B, Guanaco-7B

echo "=========================================="
echo "대용량 모델 GCG 공격 실험"
echo "GPU 24GB+ 필요!"
echo "=========================================="

NUM_SAMPLES=${1:-100}

# 대용량 모델들
MODELS=(
    "tiiuae/falcon-7b"
    "mosaicml/mpt-7b"
    "timdettmers/guanaco-7b"
)

RESULTS_DIR="results_heavy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "실험 설정:"
echo "  - 모델 수: ${#MODELS[@]}"
echo "  - 샘플 수: $NUM_SAMPLES"
echo "  - 결과 디렉토리: $RESULTS_DIR"
echo ""

CURRENT=0
TOTAL_MODELS=${#MODELS[@]}

for MODEL in "${MODELS[@]}"
do
    CURRENT=$((CURRENT + 1))
    
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL_MODELS] $MODEL"
    echo "=========================================="
    
    MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
    OUTPUT_FILE="$RESULTS_DIR/results_${MODEL_SAFE}.json"
    LOG_FILE="$RESULTS_DIR/log_${MODEL_SAFE}.txt"
    
    python simple_gcg_attack.py \
        --model_name "$MODEL" \
        --num_samples $NUM_SAMPLES \
        --use_existing_suffix \
        --output_file "$OUTPUT_FILE" \
        --log_file "$LOG_FILE" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✅ 완료!"
    else
        echo "❌ 실패"
    fi
    
    # GPU 메모리 정리 대기
    sleep 10
done

echo ""
echo "=========================================="
echo "실험 완료!"
echo "=========================================="
python compare_all_results.py "$RESULTS_DIR"

