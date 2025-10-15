#!/bin/bash

# MPT-7B 단독 실험 스크립트

echo "=========================================="
echo "MPT-7B 단독 GCG 공격 실험"
echo "=========================================="

NUM_SAMPLES=${1:-100}
MODEL="mosaicml/mpt-7b"

echo ""
echo "모델: $MODEL"
echo "샘플 수: $NUM_SAMPLES"
echo ""

# 결과 디렉토리 (기존 results에 추가 또는 새로 생성)
if [ -z "$2" ]; then
    RESULTS_DIR="results_mpt7b_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"
else
    RESULTS_DIR="$2"
    echo "기존 결과 디렉토리에 추가: $RESULTS_DIR"
fi

MODEL_SAFE="mosaicml_mpt-7b"
OUTPUT_FILE="$RESULTS_DIR/results_${MODEL_SAFE}.json"
LOG_FILE="$RESULTS_DIR/log_${MODEL_SAFE}.txt"

echo "출력 파일: $OUTPUT_FILE"
echo "로그 파일: $LOG_FILE"
echo ""
echo "시작 시간: $(date)"
echo ""

# 실험 실행
python simple_gcg_attack.py \
    --model_name "$MODEL" \
    --num_samples $NUM_SAMPLES \
    --use_existing_suffix \
    --output_file "$OUTPUT_FILE" \
    --log_file "$LOG_FILE" \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ MPT-7B 실험 완료!"
    echo ""
    
    # 결과 출력
    if [ -f "$OUTPUT_FILE" ]; then
        python -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
    print(f\"성공률: {data['success_rate']:.2f}% ({data['success_count']}/{data['num_samples']})\")
"
    fi
    
    echo ""
    echo "결과 파일: $OUTPUT_FILE"
    echo "로그 파일: $LOG_FILE"
    
    # 다른 모델들과 비교 (기존 디렉토리에 추가한 경우)
    if [ -f "$RESULTS_DIR/comparison.csv" ] || ls "$RESULTS_DIR"/results_*.json >/dev/null 2>&1; then
        echo ""
        echo "=========================================="
        echo "전체 모델 결과 업데이트"
        echo "=========================================="
        python compare_all_results.py "$RESULTS_DIR"
    fi
else
    echo ""
    echo "❌ MPT-7B 실험 실패"
    echo "로그를 확인하세요: $LOG_FILE"
fi

echo ""
echo "종료 시간: $(date)"
echo "=========================================="

