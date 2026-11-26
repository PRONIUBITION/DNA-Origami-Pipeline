# Environment Setup

Create a Python environment using the Python version supported by the
installed PyTorch build.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../stage2_hrnet/requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

Install the official YOLOv5 repository separately and set:

```text
YOLOV5_DIR=/path/to/yolov5
YOLO_WEIGHTS=/path/to/yolo_best.pt
HRNET_WEIGHTS=/path/to/hrnet_best.pth
INPUT_DIR=/path/to/input-images
OUTPUT_DIR=/path/to/results
```

Verify the environment:

```bash
python -c "import torch, cv2, yaml; print(torch.__version__)"
python ../datapredict/run.py
```
