@echo off
cd /d "%~dp0..\stage2_hrnet"
python train.py --type 00
