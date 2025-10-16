#!/bin/bash

# 각 모델당:
# - 기존 suffix로 100개 테스트 (신뢰성 높은 평가)
# - 새 suffix 생성 10개 (충분한 시도)

echo "=========================================="
echo "종합 평가: 기존 100개 + 생성 10개"
echo "=========================================="

NUM_SAMPLES_EXISTING=${1:-100}
NUM_SAMPLES_GENERATED=${2:-10}

MODELS=(
    "gpt2"
    "gpt2-medium"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "tiiuae/falcon-7b"
    "mosaicml/mpt-7b"
)

RESULTS_DIR="results_100plus10_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "실험 설정:"
echo "  - 모델 수: ${#MODELS[@]}"
echo "  - 기존 suffix: $NUM_SAMPLES_EXISTING 샘플"
echo "  - 생성 suffix: $NUM_SAMPLES_GENERATED 샘플 (각 모델당)"
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
    
    # 1. 기존 suffix (100개)
    echo ""
    echo "[1/2] 기존 Suffix 평가 ($NUM_SAMPLES_EXISTING 샘플)"
    OUTPUT_FILE_EXISTING="$RESULTS_DIR/results_${MODEL_SAFE}_existing.json"
    LOG_FILE_EXISTING="$RESULTS_DIR/log_${MODEL_SAFE}_existing.txt"
    
    python run_comprehensive_with_generation.py \
        --model_name "$MODEL" \
        --num_samples $NUM_SAMPLES_EXISTING \
        --output_file "$OUTPUT_FILE_EXISTING" \
        --log_file "$LOG_FILE_EXISTING" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✅ 기존 suffix 평가 완료"
        
        if [ -f "$OUTPUT_FILE_EXISTING" ]; then
            python -c "
import json
with open('$OUTPUT_FILE_EXISTING', 'r') as f:
    data = json.load(f)
    stats = data['statistics']
    print(f\"  Jailbreak: {stats['jailbreak_hybrid']['rate']:.2f}%\")
    print(f\"  Harmful: {stats['harmful_responses']['rate']:.2f}%\")
    print(f\"  Avg Harm: {stats['average_harm_score']:.3f}\")
"
        fi
    else
        echo "❌ 기존 suffix 평가 실패"
    fi
    
    sleep 3
    
    # 2. 새 suffix 생성
    echo ""
    echo "[2/2] 새 Suffix 생성 (10개 샘플)"
    OUTPUT_FILE_GENERATED="$RESULTS_DIR/results_${MODEL_SAFE}_generated.json"
    LOG_FILE_GENERATED="$RESULTS_DIR/log_${MODEL_SAFE}_generated.txt"
    
    python run_comprehensive_with_generation.py \
        --model_name "$MODEL" \
        --num_samples $NUM_SAMPLES_GENERATED \
        --generate_suffix \
        --num_steps 500 \
        --suffix_length 20 \
        --output_file "$OUTPUT_FILE_GENERATED" \
        --log_file "$LOG_FILE_GENERATED" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✅ 새 suffix 생성 완료"
        
        if [ -f "$OUTPUT_FILE_GENERATED" ]; then
            python -c "
import json
with open('$OUTPUT_FILE_GENERATED', 'r') as f:
    data = json.load(f)
    stats = data['statistics']
    gen = data['generation_params']
    results = data['results']
    print(f\"  생성 시간: {gen['total_generation_time']:.1f}초\")
    if results:
        print(f\"  생성 Suffix: {results[0].get('suffix', 'N/A')[:50]}...\")
        print(f\"  Jailbreak: {'✅' if results[0]['jailbreak']['hybrid'] else '❌'}\")
        print(f\"  Harm Score: {results[0]['harm']['harm_score']:.3f}\")
        print(f\"  Risk: {results[0]['overall_risk']}\")
"
        fi
    else
        echo "❌ 새 suffix 생성 실패"
    fi
    
    echo "종료 시간: $(date)"
    sleep 5
done

# 결과 요약
echo ""
echo "=========================================="
echo "전체 평가 완료!"
echo "=========================================="
echo ""

python summarize_100plus1_results.py "$RESULTS_DIR"

echo ""
echo "결과 디렉토리: $RESULTS_DIR"
echo "=========================================="

