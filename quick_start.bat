@echo off
echo Starting MF Advisor Training Pipeline...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Check GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo.
echo Choose an option:
echo 1. Run complete pipeline (data collection + training)
echo 2. Collect historical data only
echo 3. Generate training data only
echo 4. Train model only
echo 5. Start API server
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    python run_complete_training.py
) else if "%choice%"=="2" (
    python -m src.historical_data_collector
) else if "%choice%"=="3" (
    python -m src.generate_gemma_training_data
) else if "%choice%"=="4" (
    python -m src.train_gemma_gpu
) else if "%choice%"=="5" (
    python api/simple_app.py
) else (
    echo Invalid choice!
)

pause