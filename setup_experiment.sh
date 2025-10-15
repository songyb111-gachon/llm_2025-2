#!/bin/bash

# GCG 실험 환경 완전 설정 스크립트
# 패키지 설치 + 데이터 다운로드 + 모델 다운로드

echo "=========================================="
echo "GCG 실험 환경 설정"
echo "=========================================="

# 1. 패키지 설치 확인
echo ""
echo "[1/3] 패키지 확인..."
python -c "import torch; import transformers; print('✓ 패키지 설치됨')" 2>/dev/null || {
    echo "패키지를 설치합니다..."
    pip install -r requirements.txt
}

# 2. Attack artifacts 다운로드
echo ""
echo "[2/3] Attack artifacts 다운로드..."
if [ -f "vicuna-13b-v1.5.json" ]; then
    echo "✓ Artifacts 이미 존재"
else
    python download_artifacts.py
fi

# 3. 모델 다운로드
echo ""
echo "[3/3] 모델 다운로드..."
echo "다운로드할 모델 크기를 선택하세요:"
echo "  1) small  - gpt2, gpt2-medium 등 (빠름, GPU 8GB)"
echo "  2) medium - pythia-1.4b, pythia-2.8b 등 (추천, GPU 16GB)"
echo "  3) large  - falcon-7b, mpt-7b 등 (느림, GPU 24GB+)"
echo "  4) 건너뛰기"

read -p "선택 (1-4) [기본: 2]: " choice
choice=${choice:-2}

case $choice in
    1)
        python download_models.py --size small
        ;;
    2)
        python download_models.py --size medium
        ;;
    3)
        python download_models.py --size large
        ;;
    4)
        echo "모델 다운로드 건너뜀"
        ;;
    *)
        echo "잘못된 선택. 건너뜀"
        ;;
esac

# 완료
echo ""
echo "=========================================="
echo "설정 완료!"
echo "=========================================="
echo ""
echo "다음 명령어로 실험을 시작하세요:"
echo "  bash run_with_existing_suffix.sh \"gpt2\" 100"
echo "  bash run_with_existing_suffix.sh \"EleutherAI/pythia-1.4b\" 100"
echo ""
echo "또는 빠른 테스트:"
echo "  python quick_test.py gpt2"
echo "=========================================="

