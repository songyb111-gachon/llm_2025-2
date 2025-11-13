@echo off
REM Quick test for model extraction attack

echo ========================================
echo QUICK EXTRACTION ATTACK TEST
echo ========================================
echo.
echo This will run a quick test with:
echo - 100 training samples
echo - 50 test samples
echo - 1 epoch
echo - Small batch size
echo.
echo This should take only a few minutes.
echo.

python quick_extraction_test.py

pause

