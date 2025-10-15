#!/bin/bash

# 실험에 사용할 모든 모델을 미리 다운로드하는 스크립트

echo "=========================================="
echo "GCG 실험용 모델 일괄 다운로드"
echo "=========================================="
echo ""
echo "다운로드할 모델 세트를 선택하세요:"
echo "  1) experiment-all   - 전체 실험 모델 (6개) - gpt2, pythia-1.4b/2.8b, falcon-7b, mpt-7b"
echo "  2) experiment-light - 경량 모델만 (4개) - gpt2, pythia-1.4b/2.8b"
echo "  3) experiment-heavy - 대용량 모델 (3개) - falcon-7b, mpt-7b, guanaco-7b"
echo "  4) 취소"
echo ""

read -p "선택 (1-4) [기본: 2]: " choice
choice=${choice:-2}

case $choice in
    1)
        echo ""
        echo "전체 실험 모델 다운로드 시작..."
        python download_models.py --preset experiment-all
        ;;
    2)
        echo ""
        echo "경량 실험 모델 다운로드 시작..."
        python download_models.py --preset experiment-light
        ;;
    3)
        echo ""
        echo "대용량 실험 모델 다운로드 시작..."
        python download_models.py --preset experiment-heavy
        ;;
    4)
        echo "취소됨"
        exit 0
        ;;
    *)
        echo "잘못된 선택"
        exit 1
        ;;
esac

# 다운로드 완료 후 확인
echo ""
echo "=========================================="
echo "다운로드 완료!"
echo "=========================================="
echo ""
echo "다운로드된 모델 확인:"
python check_models.py

echo ""
echo "이제 실험을 시작할 수 있습니다:"
echo "  bash run_all_models.sh 100          # 전체 모델"
echo "  bash run_all_models_light.sh 100    # 경량 모델"
echo "  bash run_all_models_heavy.sh 100    # 대용량 모델"
echo "=========================================="

