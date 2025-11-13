@echo off
echo ================================================
echo 3단계: 결과 시각화
echo ================================================
echo.
echo 그래프를 생성합니다.
echo visualizations/ 폴더에 PNG 파일이 저장됩니다.
echo.
pause

python visualize_results.py --summary results/experiment_summary.json --output visualizations

echo.
echo ================================================
echo 완료! visualizations/ 폴더를 확인하세요.
echo ================================================
pause



