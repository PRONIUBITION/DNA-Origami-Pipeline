@echo off
cd /d "%~dp0..\..\third_party\yolov5"
if errorlevel 1 (echo YOLOv5 directory not found under third_party\yolov5. & pause & exit /b 1)
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_WAIT_POLICY=PASSIVE
if "%YOLO_WEIGHTS%"=="" set YOLO_WEIGHTS=yolov5s-seg.pt
if "%DATA_YAML%"=="" set DATA_YAML=%~dp0..\stage1_yolo\configs\polymer.example.yaml
if "%HYP_YAML%"=="" set HYP_YAML=%~dp0..\stage1_yolo\configs\hyp.polymer.yaml
python segment/train.py --weights "%YOLO_WEIGHTS%" --data "%DATA_YAML%" --hyp "%HYP_YAML%" --batch-size 2 --workers 0 --epochs 500 --patience 0 --img 640 --name polymer_dimmer --project runs\train-seg --exist-ok
