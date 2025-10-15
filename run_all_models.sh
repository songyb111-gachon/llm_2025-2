#!/bin/bash

# 모든 추천 모델에 대해 GCG 공격 실험 실행
# 100개 샘플로 성공률 측정

echo "=========================================="
echo "전체 모델 GCG 공격 실험"
echo "=========================================="

# 실험 설정
NUM_SAMPLES=${1:-100}
USE_EXISTING_SUFFIX=${2:-true}

echo ""
echo "실험 설정:"
echo "  - 샘플 수: $NUM_SAMPLES"
echo "  - 기존 suffix 사용: $USE_EXISTING_SUFFIX"
echo ""

# 실험할 모델 리스트
MODELS=(
    "gpt2"
    "gpt2-medium"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "tiiuae/falcon-7b"
    "mosaicml/mpt-7b"
)

# 결과 디렉토리 생성
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "결과 저장 디렉토리: $RESULTS_DIR"
echo ""

# 총 모델 수
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0
SUCCESS_MODELS=()
FAILED_MODELS=()

# 각 모델에 대해 실험 실행
for MODEL in "${MODELS[@]}"
do
    CURRENT=$((CURRENT + 1))
    
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL_MODELS] 모델: $MODEL"
    echo "=========================================="
    
    # 모델 이름에서 '/'를 '_'로 변경 (파일명용)
    MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
    OUTPUT_FILE="$RESULTS_DIR/results_${MODEL_SAFE}.json"
    LOG_FILE="$RESULTS_DIR/log_${MODEL_SAFE}.txt"
    
    echo "시작 시간: $(date)"
    echo "출력 파일: $OUTPUT_FILE"
    echo ""
    
    # 실험 실행
    if [ "$USE_EXISTING_SUFFIX" = true ]; then
        # 기존 suffix 사용 (빠름)
        python simple_gcg_attack.py \
            --model_name "$MODEL" \
            --num_samples $NUM_SAMPLES \
            --use_existing_suffix \
            --output_file "$OUTPUT_FILE" \
            --log_file "$LOG_FILE" \
            --device cuda
    else
        # 새로운 suffix 생성 (느림)
        python simple_gcg_attack.py \
            --model_name "$MODEL" \
            --num_samples $NUM_SAMPLES \
            --num_steps 250 \
            --output_file "$OUTPUT_FILE" \
            --log_file "$LOG_FILE" \
            --device cuda
    fi
    
    # 결과 확인
    if [ $? -eq 0 ]; then
        SUCCESS_MODELS+=("$MODEL")
        echo ""
        echo "✅ $MODEL 실험 완료!"
        
        # 간단한 결과 출력
        if [ -f "$OUTPUT_FILE" ]; then
            python -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
    print(f\"성공률: {data['success_rate']:.2f}% ({data['success_count']}/{data['num_samples']})\")
"
        fi
    else
        FAILED_MODELS+=("$MODEL")
        echo ""
        echo "❌ $MODEL 실험 실패"
    fi
    
    echo "종료 시간: $(date)"
    echo ""
    
    # GPU 메모리 정리 대기
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

# 결과 비교 표 생성
echo "=========================================="
echo "모델별 성공률 비교"
echo "=========================================="
echo ""

python compare_all_results.py "$RESULTS_DIR"

echo ""
echo "결과 디렉토리: $RESULTS_DIR"
echo "=========================================="

