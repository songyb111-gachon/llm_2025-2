#!/bin/bash

# 기존 결과 디렉토리에 MPT-7B 결과 추가

echo "=========================================="
echo "기존 결과에 MPT-7B 추가"
echo "=========================================="
echo ""

# 결과 디렉토리 찾기
if [ -z "$1" ]; then
    # 가장 최근 results 디렉토리 자동 찾기
    RESULTS_DIR=$(ls -td results_*/ 2>/dev/null | head -1)
    
    if [ -z "$RESULTS_DIR" ]; then
        echo "❌ 결과 디렉토리를 찾을 수 없습니다."
        echo ""
        echo "사용법:"
        echo "  bash add_mpt7b_to_results.sh [결과_디렉토리]"
        echo ""
        echo "예시:"
        echo "  bash add_mpt7b_to_results.sh results_20241015_123456"
        exit 1
    fi
    
    echo "자동 감지된 결과 디렉토리: $RESULTS_DIR"
else
    RESULTS_DIR="$1"
    
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "❌ 디렉토리가 존재하지 않습니다: $RESULTS_DIR"
        exit 1
    fi
fi

NUM_SAMPLES=${2:-100}

echo "결과 디렉토리: $RESULTS_DIR"
echo "샘플 수: $NUM_SAMPLES"
echo ""

# 기존 결과 확인
echo "기존 모델 결과:"
ls -1 "$RESULTS_DIR"/results_*.json 2>/dev/null | sed 's/.*results_/  - /' | sed 's/.json$//'
echo ""

read -p "MPT-7B를 이 디렉토리에 추가하시겠습니까? (y/n) [y]: " confirm
confirm=${confirm:-y}

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "취소됨"
    exit 0
fi

echo ""
echo "MPT-7B 실험 시작..."
bash run_mpt7b.sh "$NUM_SAMPLES" "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "완료!"
echo "=========================================="
echo ""
echo "전체 결과 확인:"
echo "  cat $RESULTS_DIR/report.txt"
echo "  cat $RESULTS_DIR/comparison.csv"

