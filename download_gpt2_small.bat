@echo off
echo ================================================
echo GPT-2 작은 모델만 다운로드 (16GB GPU 이하)
echo ================================================
echo.
echo 다운로드할 모델:
echo - distilgpt2 (88M)
echo - gpt2 (124M)
echo - gpt2-large (774M)
echo.
echo 시작하려면 아무 키나 누르세요...
pause > nul

python download_gpt2_models.py --small-only

echo.
echo ================================================
echo 완료!
echo ================================================
pause

