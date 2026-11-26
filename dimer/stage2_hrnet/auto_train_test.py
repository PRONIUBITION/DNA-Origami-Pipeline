import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
CONFIG = ROOT / "configs" / "config.yaml"
RUNS = ROOT / "runs"
VAL_RESULTS = ROOT / "outputs" / "val_results"
TEST_INPUT = ROOT / "data" / "processed" / "test"
MAX_ROUNDS = 20


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def current_epochs():
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(r"^\s*epochs:\s*(\d+)", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("epochs not found in config")
    return int(match.group(1))


def increase_epochs(delta=100):
    text = CONFIG.read_text(encoding="utf-8")

    def replace(match):
        prefix = match.group(1)
        value = int(match.group(2)) + delta
        suffix = match.group(3)
        return f"{prefix}{value}{suffix}"

    new_text = re.sub(
        r"(?m)^(\s*epochs:\s*)(\d+)(.*)$",
        replace,
        text,
    )
    CONFIG.write_text(new_text, encoding="utf-8")


def latest_run_dir():
    dirs = [p for p in RUNS.glob("train_*") if p.is_dir()]
    if not dirs:
        raise RuntimeError("no training run found")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def four_dot_rate(val_dir):
    results_path = val_dir / "val_results.json"
    if not results_path.exists():
        raise RuntimeError(f"missing val results: {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not results:
        raise RuntimeError("val results are empty")
    four_dots = sum(1 for r in results if int(r.get("num_dots", 0)) == 4)
    return four_dots / len(results)


def run_training():
    log(f"start training with epochs={current_epochs()}")
    proc = subprocess.run(
        [PYTHON, "train.py", "--config", str(CONFIG), "--device", "cuda"],
        cwd=str(ROOT),
    )
    log(f"train subprocess exited with code={proc.returncode}")
    run_dir = latest_run_dir()
    final_weights = run_dir / "weights" / "final_model.pth"
    if not final_weights.exists():
        raise RuntimeError(f"final weights not found: {final_weights}")
    return run_dir, final_weights


def run_test(final_weights):
    log("start prediction with final weights")
    subprocess.run(
        [
            PYTHON,
            "predict.py",
            "--weights",
            str(final_weights),
            "--config",
            str(CONFIG),
            "--input",
            str(TEST_INPUT),
            "--val",
            "--device",
            "cuda",
        ],
        cwd=str(ROOT),
    )
    train_name = final_weights.parent.parent.name
    val_dirs = [p for p in VAL_RESULTS.glob(f"val_{train_name}_*") if p.is_dir()]
    if not val_dirs:
        raise RuntimeError("no val output directory found")
    val_dir = max(val_dirs, key=lambda p: p.stat().st_mtime)
    return val_dir


def main():
    for round_index in range(1, MAX_ROUNDS + 1):
        log(f"round {round_index}")
        run_dir, final_weights = run_training()
        val_dir = run_test(final_weights)
        rate = four_dot_rate(val_dir)
        log(f"result: run={run_dir.name}, four_dot_rate={rate:.4f}, val_dir={val_dir.name}")
        if rate >= 1.0:
            log("four dot rate reached 100%; stop")
            (ROOT / "auto_train_test_done.txt").write_text(
                f"done round={round_index}\n"
                f"run={run_dir}\n"
                f"val_dir={val_dir}\n"
                f"four_dot_rate={rate:.6f}\n",
                encoding="utf-8",
            )
            return 0
        increase_epochs(100)
        log(f"increase epochs to {current_epochs()}")

    log("max rounds reached without 100% four dot rate")
    return 2


if __name__ == "__main__":
    sys.exit(main())
