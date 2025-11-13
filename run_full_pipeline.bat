@echo off
echo ================================================
echo Backdoor 실험 전체 파이프라인
echo ================================================
echo.
echo 다음 순서로 실행됩니다:
echo 1. Poisoning 데이터 생성
echo 2. 빠른 실험 (distilgpt2, 200개)
echo 3. 결과 시각화
echo.
echo 전체 소요 시간: 약 15-25분
echo.
pause

echo.
echo ================================================
echo [1/3] Poisoning 데이터 생성
echo ================================================
python generate_poisoning_data.py --all

if %ERRORLEVEL% NEQ 0 (
    echo 에러 발생! 중단합니다.
    pause
    exit /b 1
)

echo.
echo ================================================
echo [2/3] 빠른 실험 실행
echo ================================================
python run_experiments.py --quick

if %ERRORLEVEL% NEQ 0 (
    echo 에러 발생! 중단합니다.
    pause
    exit /b 1
)

echo.
echo ================================================
echo [3/3] 결과 시각화
echo ================================================
python visualize_results.py --summary results/quick_test_distilgpt2_200.json --output visualizations

echo.
echo ================================================
echo 전체 파이프라인 완료!
echo ================================================
echo.
echo 결과 확인:
echo - results/ : JSON 결과 파일
echo - visualizations/ : 그래프 (PNG)
echo - backdoored_models/ : Fine-tuned 모델
echo.
pause




