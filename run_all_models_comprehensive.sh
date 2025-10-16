#!/bin/bash

# 6개 모델 종합 평가 (JailbreakBench + HarmBench)

echo "=========================================="
echo "종합 평가: JailbreakBench + HarmBench"
echo "6개 모델 전체 실험"
echo "=========================================="

NUM_SAMPLES=${1:-100}

MODELS=(
    "gpt2"
    "gpt2-medium"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "tiiuae/falcon-7b"
    "mosaicml/mpt-7b"
)

RESULTS_DIR="results_comprehensive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "실험 설정:"
echo "  - 모델 수: ${#MODELS[@]}"
echo "  - 샘플 수: $NUM_SAMPLES"
echo "  - 평가 항목:"
echo "    ① Jailbreak (Simple/Strict/Hybrid)"
echo "    ② Harm Score (0-1)"
echo "    ③ Risk Level (CRITICAL/HIGH/MEDIUM/LOW/SAFE)"
echo "  - 결과 디렉토리: $RESULTS_DIR"
echo ""

CURRENT=0
TOTAL_MODELS=${#MODELS[@]}
SUCCESS_MODELS=()

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
    
    python run_comprehensive_evaluation.py \
        --model_name "$MODEL" \
        --num_samples $NUM_SAMPLES \
        --output_file "$OUTPUT_FILE" \
        --log_file "$LOG_FILE" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        SUCCESS_MODELS+=("$MODEL")
        echo ""
        echo "✅ $MODEL 평가 완료!"
        
        # 요약 출력
        if [ -f "$OUTPUT_FILE" ]; then
            python -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
    stats = data['statistics']
    print(f\"  Jailbreak (Hybrid): {stats['jailbreak_hybrid']['rate']:.2f}%\")
    print(f\"  Harmful: {stats['harmful_responses']['rate']:.2f}%\")
    print(f\"  Avg Harm Score: {stats['average_harm_score']:.3f}\")
    print(f\"  CRITICAL Risk: {stats['risk_distribution']['CRITICAL']}\")
"
        fi
    else
        echo ""
        echo "❌ $MODEL 평가 실패"
    fi
    
    echo "종료 시간: $(date)"
    sleep 5
done

# 최종 비교
echo ""
echo "=========================================="
echo "전체 평가 완료!"
echo "=========================================="
echo ""

python compare_comprehensive_results.py "$RESULTS_DIR"

echo ""
echo "결과 디렉토리: $RESULTS_DIR"
echo "=========================================="

