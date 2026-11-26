@echo off
chcp 65001 >nul
echo ========================================
echo DNA Origami White Dot Detection Pipeline
echo ========================================
echo.

cd /d "%~dp0..\.."

if "%YOLO_WEIGHTS%"=="" set YOLO_WEIGHTS=models\yolo_best.pt
if "%HRNET_WEIGHTS%"=="" set HRNET_WEIGHTS=models\hrnet_best.pth
if "%INPUT_DIR%"=="" set INPUT_DIR=dimer\datapredict\input
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=dimer\datapredict\outputs

echo Config:
echo   YOLO weights: %YOLO_WEIGHTS%
echo   HRNet weights: %HRNET_WEIGHTS%
echo   Input: %INPUT_DIR%
echo   Output: %OUTPUT_DIR%
echo.

echo Starting pipeline with direct Python path...
echo ========================================

set CONDA_PYTHON=python
echo Using Python: %CONDA_PYTHON%
echo.

%CONDA_PYTHON% dimer\pipeline.py ^
    --yolo-weights %YOLO_WEIGHTS% ^
    --hrnet-weights %HRNET_WEIGHTS% ^
    --input %INPUT_DIR% ^
    --output %OUTPUT_DIR%

echo.
echo ========================================
if errorlevel 1 (
    echo [ERROR] Pipeline failed!
    echo Please check error messages above.
    echo.
    echo Alternative: Run in Anaconda Prompt manually:
    echo   conda activate dnaorigami
    echo   cd /d "%~dp0..\.."
    echo   python dimer/datapredict/run.py
) else (
    echo [SUCCESS] Pipeline completed!
    echo Results saved to: %OUTPUT_DIR%
)
echo ========================================

pause
