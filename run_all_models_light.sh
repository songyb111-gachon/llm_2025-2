#!/bin/bash

# 가벼운 모델들만 실험 (GPU 메모리 부족 시)
# GPT-2, Pythia-1.4B, Pythia-2.8B

echo "=========================================="
echo "경량 모델 GCG 공격 실험"
echo "=========================================="

NUM_SAMPLES=${1:-100}

# 실험할 모델 (가벼운 것만)
MODELS=(
    "gpt2"
    "gpt2-medium"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
)

RESULTS_DIR="results_light_$(date +%Y%m%d_%H%M%S)"
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
    
    python simple_gcg_attack.py \
        --model_name "$MODEL" \
        --num_samples $NUM_SAMPLES \
        --use_existing_suffix \
        --output_file "$OUTPUT_FILE" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✅ 완료!"
    else
        echo "❌ 실패"
    fi
    
    sleep 3
done

echo ""
echo "=========================================="
echo "실험 완료!"
echo "=========================================="
python compare_all_results.py "$RESULTS_DIR"

