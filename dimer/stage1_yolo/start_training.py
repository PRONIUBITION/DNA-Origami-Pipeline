# start_training_fixed.py
import os
import sys
import subprocess
from pathlib import Path
from settings import PROJECT_ROOT, YOLOV5_DIR, YOLO_WEIGHTS


def start_training():
    """One-Click Training Start (Fix Windows Path Issues)"""

    # Configuration
    config = {
        "yolov5_dir": YOLOV5_DIR,
        "yaml_path": PROJECT_ROOT / "stage1_yolo" / "configs" / "data.example.yaml",
        "hyp_path": PROJECT_ROOT / "stage1_yolo" / "configs" / "hyp.dna.yaml",
        "weights_path": YOLO_WEIGHTS,
        "experiment_name": "gray_05",
    }

    print("""Starting to train the DNA cross detection model...""")
    print(f"""Configuration file: {config["yaml_path"]}""")
    print(f"""Experiment Name: {config["experiment_name"]}""")

    # Check the YOLOv5 directory
    if not os.path.exists(config["yolov5_dir"]):
        print(f"""YOLOv5 directory does not exist: {config["yolov5_dir"]}""")
        return

    # Check the YAML file
    if not os.path.exists(config["yaml_path"]):
        print(f"""YAML file does not exist: {config["yaml_path"]}""")

        return

    # Switch to the YOLOv5 directory
    original_dir = os.getcwd()
    os.chdir(config["yolov5_dir"])

    print(f"""\n Switch to directory: {os.getcwd()}""")

    # Build command parameter list
    cmd_args = [
        sys.executable,  # Use the current Python
        "segment/train.py",
        f"--data={config['yaml_path']}",  # Use the = sign instead of a space
        f"--weights={config['weights_path']}",
        f"--name={config['experiment_name']}",
        "--project=runs/train-seg",
        "--exist-ok",
        "--img=640",
        "--batch-size=8",
        "--epochs=500",
        "--freeze=10",
        "--patience=30",
        f"--hyp={config['hyp_path']}",
    ]

    print("""\n Execute command:""")
    print(" ".join(cmd_args))

    input("""\n Press Enter to start training...""")

    try:
        # Run using subprocess to avoid cmd parsing issues
        result = subprocess.run(cmd_args, check=True)
        print("""\n Training completed!""")

    except subprocess.CalledProcessError as e:
        print(f"""\n Training failed: {e}""")
    except KeyboardInterrupt:
        print("""\n Training interrupted""")
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    start_training()
