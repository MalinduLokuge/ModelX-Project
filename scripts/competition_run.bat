@echo off
REM Competition run (2 hours, maximum performance)
REM Usage: competition_run.bat <train_path> <test_path>

echo ========================================
echo CompeteML Competition Run (2 hours)
echo Maximum Performance Mode
echo ========================================

if "%1"=="" (
    echo Usage: competition_run.bat ^<train_path^> ^<test_path^>
    echo Example: competition_run.bat data/raw/train.csv data/raw/test.csv
    exit /b 1
)

if "%2"=="" (
    echo Usage: competition_run.bat ^<train_path^> ^<test_path^>
    echo Test path required for competition mode
    exit /b 1
)

echo Train: %1
echo Test: %2
echo.

python main.py ^
    --train %1 ^
    --test %2 ^
    --config configs/competition.yaml

echo.
echo ========================================
echo Competition run complete!
echo Check outputs/ folder for:
echo   - submission.csv (ready to upload)
echo   - recipe.txt (what was done)
echo   - model files
echo ========================================
