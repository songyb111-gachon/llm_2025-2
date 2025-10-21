@echo off
echo ================================================
echo GPT-2 계열 모델 전체 다운로드
echo ================================================
echo.
echo 다운로드할 모델:
echo - distilgpt2 (88M)
echo - gpt2 (124M)
echo - gpt2-large (774M)
echo - gpt2-xl (1.5B)
echo.
echo 시작하려면 아무 키나 누르세요...
pause > nul

python download_gpt2_models.py --all

echo.
echo ================================================
echo 완료!
echo ================================================
pause

