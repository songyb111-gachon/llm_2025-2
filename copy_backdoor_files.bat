@echo off
echo ================================================
echo Backdoor 실험 파일들을 새 폴더로 복사
echo ================================================
echo.

set SOURCE=.
set DEST=backdoor-sleeper-agents

echo 복사 시작...
echo.

:: Python 스크립트
echo [1/4] Python 스크립트 복사 중...
copy "%SOURCE%\generate_poisoning_data.py" "%DEST%\" > nul
copy "%SOURCE%\finetune_backdoor.py" "%DEST%\" > nul
copy "%SOURCE%\evaluate_backdoor.py" "%DEST%\" > nul
copy "%SOURCE%\run_experiments.py" "%DEST%\" > nul
copy "%SOURCE%\visualize_results.py" "%DEST%\" > nul
copy "%SOURCE%\download_gpt2_models.py" "%DEST%\" > nul
echo   Python 스크립트 6개 복사 완료!

:: 배치 파일
echo [2/4] 배치 파일 복사 중...
copy "%SOURCE%\1_generate_data.bat" "%DEST%\" > nul
copy "%SOURCE%\2_quick_experiment.bat" "%DEST%\" > nul
copy "%SOURCE%\3_visualize.bat" "%DEST%\" > nul
copy "%SOURCE%\run_full_pipeline.bat" "%DEST%\" > nul
copy "%SOURCE%\download_gpt2_all.bat" "%DEST%\" > nul
copy "%SOURCE%\download_gpt2_small.bat" "%DEST%\" > nul
echo   배치 파일 6개 복사 완료!

:: 문서
echo [3/4] 문서 복사 중...
copy "%SOURCE%\BACKDOOR_EXPERIMENT_README.md" "%DEST%\EXPERIMENT_GUIDE.md" > nul
copy "%SOURCE%\BACKDOOR_QUICKSTART.md" "%DEST%\QUICKSTART.md" > nul
copy "%SOURCE%\BACKDOOR_PROJECT_SUMMARY.md" "%DEST%\PROJECT_SUMMARY.md" > nul
echo   문서 3개 복사 완료!

:: 설정 파일
echo [4/4] 설정 파일 생성 중...
:: requirements.txt는 별도로 생성

echo.
echo ================================================
echo 복사 완료!
echo ================================================
echo.
echo 다음 파일들이 복사되었습니다:
echo.
echo backdoor-sleeper-agents/
echo   ├── Python 스크립트 (6개)
echo   ├── 배치 파일 (6개)
echo   ├── 문서 (4개 - README 포함)
echo   └── requirements.txt
echo.
echo 다음 명령어로 Git 초기화:
echo   cd backdoor-sleeper-agents
echo   git init
echo   git add .
echo   git commit -m "Initial commit: Backdoor Sleeper Agents experiment framework"
echo.
pause



