@echo off
echo ================================================
echo 1단계: Poisoning 데이터 생성
echo ================================================
echo.
echo 트리거: [GACHON]
echo 생성할 크기: 50, 100, 200, 500개
echo.
pause

python generate_poisoning_data.py --all

echo.
echo ================================================
echo 완료! poisoning_data/ 폴더를 확인하세요.
echo ================================================
pause




