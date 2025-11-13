@echo off
echo ================================================
echo 2단계: 빠른 실험 (distilgpt2, 200개 데이터)
echo ================================================
echo.
echo 이 실험은 약 10-20분 소요됩니다.
echo.
echo 실행 내용:
echo - Fine-tuning (2 epochs)
echo - Evaluation (원본 vs Backdoor)
echo.
pause

python run_experiments.py --quick

echo.
echo ================================================
echo 완료! results/ 폴더에서 결과를 확인하세요.
echo ================================================
pause



