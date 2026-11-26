@echo off
cd /d "%~dp0..\.."

:: Use Python from the active environment.
set CONDA_PYTHON=python

if not exist %CONDA_PYTHON% (
    set CONDA_PYTHON=python
)

echo Running grouping comparison...
%CONDA_PYTHON% dimer\stage2_hrnet\compare_grouping.py ^
    --weights "%HRNET_WEIGHTS%" ^
    --config "dimer\stage2_hrnet\configs\config.yaml" ^
    --input "dimer\stage2_hrnet\data\processed\val" ^
    --output "%OUTPUT_DIR%\grouping_comparison" ^
    --threshold 50

echo.
pause
