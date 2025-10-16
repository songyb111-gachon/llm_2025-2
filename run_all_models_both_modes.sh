#!/bin/bash

# 두 가지 모드로 모든 모델 평가
# 모드 1: 기존 suffix (빠름, 100개)
# 모드 2: 새 suffix 생성 (느림, 더 적은 샘플)

echo "=========================================="
echo "두 가지 모드 종합 평가"
echo "=========================================="

# 기존 suffix 샘플 수
NUM_SAMPLES_EXISTING=${1:-100}
# 새 suffix 샘플 수 (적게)
NUM_SAMPLES_GENERATED=${2:-10}

MODELS=(
    "gpt2"
    "gpt2-medium"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "tiiuae/falcon-7b"
    "mosaicml/mpt-7b"
)

RESULTS_DIR="results_both_modes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "실험 설정:"
echo "  - 모델 수: ${#MODELS[@]}"
echo "  - 모드 1 (기존 suffix): $NUM_SAMPLES_EXISTING 샘플"
echo "  - 모드 2 (생성 suffix): $NUM_SAMPLES_GENERATED 샘플"
echo "  - 결과 디렉토리: $RESULTS_DIR"
echo ""

for MODEL in "${MODELS[@]}"
do
    echo ""
    echo "=========================================="
    echo "모델: $MODEL"
    echo "=========================================="
    
    MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
    
    # 모드 1: 기존 suffix (빠름)
    echo ""
    echo "[모드 1] 기존 Suffix 평가 ($NUM_SAMPLES_EXISTING 샘플)"
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
    print(f\"  Harm Score: {stats['average_harm_score']:.3f}\")
"
        fi
    else
        echo "❌ 기존 suffix 평가 실패"
    fi
    
    sleep 3
    
    # 모드 2: 새 suffix 생성 (느림)
    echo ""
    echo "[모드 2] 새 Suffix 생성 ($NUM_SAMPLES_GENERATED 샘플)"
    OUTPUT_FILE_GENERATED="$RESULTS_DIR/results_${MODEL_SAFE}_generated.json"
    LOG_FILE_GENERATED="$RESULTS_DIR/log_${MODEL_SAFE}_generated.txt"
    
    python run_comprehensive_with_generation.py \
        --model_name "$MODEL" \
        --num_samples $NUM_SAMPLES_GENERATED \
        --generate_suffix \
        --num_steps 250 \
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
    print(f\"  Jailbreak: {stats['jailbreak_hybrid']['rate']:.2f}%\")
    print(f\"  Harmful: {stats['harmful_responses']['rate']:.2f}%\")
    print(f\"  Harm Score: {stats['average_harm_score']:.3f}\")
    print(f\"  평균 생성시간: {gen['total_generation_time']/data['num_samples']:.1f}초/샘플\")
"
        fi
    else
        echo "❌ 새 suffix 생성 실패"
    fi
    
    echo "종료 시간: $(date)"
    sleep 5
done

# 결과 비교
echo ""
echo "=========================================="
echo "전체 평가 완료!"
echo "=========================================="
echo ""

python compare_both_modes_results.py "$RESULTS_DIR"

echo ""
echo "결과 디렉토리: $RESULTS_DIR"
echo "=========================================="

