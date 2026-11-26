"""Monomer YOLOv5-seg ()"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from settings import YOLOV5_DIR, YOLO_WEIGHTS, data_path, output_path

PYTHON = sys.executable
monomer_type = os.environ.get("MONOMER_TYPE", "00")
DATA_YAML = Path(
    os.environ.get("DATA_YAML", data_path("yolo_datasets", f"type{monomer_type}", "data.yaml"))
).resolve()
HYP_YAML = Path(os.environ.get("HYP_YAML", YOLOV5_DIR / "data" / "hyps" / "hyp.dna.yaml")).resolve()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"monomer_yolo_train_{timestamp}"
    log_dir = output_path("yolo_runs", "train-seg", run_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    print("=" * 60)
    print(f"""DNA YOLOv5-seg""")
    print(f""": {log_file}""")
    print(f""": {run_name}""")
    print("=" * 60)

    cmd = [
        str(PYTHON),
        str(YOLOV5_DIR / "segment" / "train.py"),
        f"--data={DATA_YAML}",
        f"--weights={YOLO_WEIGHTS}",
        f"--hyp={HYP_YAML}",
        "--img=480",
        "--batch-size=4",
        "--epochs=300",
        "--patience=30",
        "--freeze=0",
        f"--project={output_path('yolo_runs', 'train-seg')}",
        f"--name={run_name}",
        "--exist-ok",
    ]

    print(f"Python: {PYTHON}")
    print(f""": {DATA_YAML}""")
    print(f""": {HYP_YAML}""")
    print(f""": {log_dir}""")
    print(f"""\n : \n {" ".join(cmd)} \n""")

    # Run and log in real-time
    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(YOLOV5_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        proc.wait()

    if proc.returncode == 0:
        print(f"""\n [OK] !: {log_file}""")
    else:
        print(f"""\n [FAIL] (code= {proc.returncode} )""")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
