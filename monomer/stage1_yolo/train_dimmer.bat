@echo off
cd /d "%~dp0..\..\third_party\yolov5"
if errorlevel 1 (echo CD FAILED & pause & exit /b 1)
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_WAIT_POLICY=PASSIVE
if "%YOLO_WEIGHTS%"=="" set YOLO_WEIGHTS=yolov5s-seg.pt
if "%DATA_YAML%"=="" set DATA_YAML=..\..\configs\data.example.yaml
if "%HYP_YAML%"=="" set HYP_YAML=data\hyps\hyp.dna.yaml
python segment/train.py --weights "%YOLO_WEIGHTS%" --data "%DATA_YAML%" --hyp "%HYP_YAML%" --batch-size 4 --epochs 500 --patience 30 --img 640 --name monomer_yolo --project "runs\train-seg" --exist-ok
