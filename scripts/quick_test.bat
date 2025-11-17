@echo off
REM Quick 5-minute test run
REM Usage: quick_test.bat <data_path>

echo ========================================
echo CompeteML Quick Test (5 minutes)
echo ========================================

if "%1"=="" (
    echo Usage: quick_test.bat ^<data_path^>
    echo Example: quick_test.bat data/raw/my_data.csv
    exit /b 1
)

python main.py ^
    --train %1 ^
    --config configs/quick_test.yaml

echo.
echo ========================================
echo Quick test complete!
echo Check outputs/ folder for results
echo ========================================
