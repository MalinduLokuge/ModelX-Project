@echo off
REM Setup CompeteML environment

echo ========================================
echo Setting up CompeteML
echo ========================================

REM Install dependencies
echo.
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt

REM Create directories
echo.
echo [2/3] Creating directory structure...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "outputs\models" mkdir outputs\models
if not exist "outputs\reports" mkdir outputs\reports
if not exist "outputs\recipes" mkdir outputs\recipes
if not exist "outputs\submissions" mkdir outputs\submissions
if not exist "outputs\logs" mkdir outputs\logs

REM Verify installation
echo.
echo [3/3] Verifying installation...
python -c "import autogluon; print('AutoGluon:', autogluon.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"

echo.
echo ========================================
echo Setup complete!
echo.
echo Next steps:
echo   1. Place your data in data/raw/
echo   2. Run: scripts\quick_test.bat data/raw/your_data.csv
echo   3. Review outputs/ folder
echo ========================================
