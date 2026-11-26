@echo off
cd /d "%~dp0..\..\third_party\yolov5"
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_WAIT_POLICY=PASSIVE
python segment/train.py --resume runs\train-seg\polymer_dimmer_v5\weights\last.pt --workers 0
