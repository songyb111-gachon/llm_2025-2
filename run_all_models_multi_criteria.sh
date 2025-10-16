#!/bin/bash

# 6개 모델 × 3가지 평가 기준 전체 실험

echo "=========================================="
echo "멀티 기준 GCG 공격 실험"
echo "6개 모델 × 3가지 평가 기준"
echo "=========================================="

NUM_SAMPLES=${1:-100}

# 6개 모델 (mpt-7b 포함)
MODELS=(
    "gpt2"
    "gpt2-medium"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "tiiuae/falcon-7b"
    "mosaicml/mpt-7b"
)

RESULTS_DIR="results_multi_criteria_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "실험 설정:"
echo "  - 모델 수: ${#MODELS[@]}"
echo "  - 샘플 수: $NUM_SAMPLES"
echo "  - 평가 기준: Simple, Strict, Hybrid (3가지)"
echo "  - 결과 디렉토리: $RESULTS_DIR"
echo ""

CURRENT=0
TOTAL_MODELS=${#MODELS[@]}
SUCCESS_MODELS=()
FAILED_MODELS=()

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
    
    echo "시작 시간: $(date)"
    
    python run_multi_criteria_experiment.py \
        --model_name "$MODEL" \
        --num_samples $NUM_SAMPLES \
        --output_file "$OUTPUT_FILE" \
        --log_file "$LOG_FILE" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        SUCCESS_MODELS+=("$MODEL")
        echo ""
        echo "✅ $MODEL 실험 완료!"
        
        # 결과 출력
        if [ -f "$OUTPUT_FILE" ]; then
            python -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
    print(f\"  Simple: {data['success_rate_simple']:.2f}% ({data['success_count_simple']}/{data['num_samples']})\")
    print(f\"  Strict: {data['success_rate_strict']:.2f}% ({data['success_count_strict']}/{data['num_samples']})\")
    print(f\"  Hybrid: {data['success_rate_hybrid']:.2f}% ({data['success_count_hybrid']}/{data['num_samples']})\")
"
        fi
    else
        FAILED_MODELS+=("$MODEL")
        echo ""
        echo "❌ $MODEL 실험 실패"
    fi
    
    echo "종료 시간: $(date)"
    
    # GPU 메모리 정리
    sleep 5
done

# 최종 요약
echo ""
echo "=========================================="
echo "전체 실험 완료!"
echo "=========================================="
echo ""
echo "📊 요약:"
echo "  총 모델: $TOTAL_MODELS"
echo "  성공: ${#SUCCESS_MODELS[@]}"
echo "  실패: ${#FAILED_MODELS[@]}"
echo ""

if [ ${#SUCCESS_MODELS[@]} -gt 0 ]; then
    echo "✅ 성공한 모델:"
    for MODEL in "${SUCCESS_MODELS[@]}"; do
        echo "  - $MODEL"
    done
    echo ""
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "❌ 실패한 모델:"
    for MODEL in "${FAILED_MODELS[@]}"; do
        echo "  - $MODEL"
    done
    echo ""
fi

# 결과 비교
echo "=========================================="
echo "평가 기준별 결과 비교"
echo "=========================================="
echo ""

python compare_multi_criteria_results.py "$RESULTS_DIR"

echo ""
echo "결과 디렉토리: $RESULTS_DIR"
echo "=========================================="

