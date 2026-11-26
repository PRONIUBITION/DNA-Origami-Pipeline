"""Measure the pipeline throughput in bits/s and MB/s.

Two numbers are reported because they mean different things:

  compute throughput  segmentation + feature extraction + encoding only, with
                      the Qwen codes already available
  end to end          the above plus the Qwen API call per cross

Recognition time is recovered from the result-file timestamps written once per
crop; the median is used so retries and timeouts do not dominate.
"""

import json
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BASE = ROOT / "outputs" / "qwen_recognition"
PYTHON = __import__("sys").executable


def run(script, *args):
    start = time.perf_counter()
    subprocess.run(
        [PYTHON, str(SRC / script), *args],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - start


def recognition_times():
    """Per-crop recognition seconds, from consecutive result-file mtimes."""
    samples = []
    for group in sorted(p for p in BASE.iterdir() if p.is_dir()):
        run_dir = group / "thinking_clean"
        files = sorted(
            run_dir.glob("cross_*_results.json"),
            key=lambda p: p.stat().st_mtime,
        )
        stamps = [p.stat().st_mtime for p in files]
        samples.extend(b - a for a, b in zip(stamps[:-1], stamps[1:]) if 0 < b - a < 1800)
    return samples


def main():
    stream_stats = json.loads((BASE / "final" / "stream_stats.json").read_text(encoding="utf-8"))
    crosses = stream_stats["crosses"]
    total_bits = stream_stats["total_bits"]
    bits_per_cross = stream_stats["bits_per_cross"]

    print("timing each stage ...")
    seg = run("segment_union.py")
    feat = run(
        "build_all_features.py",
        "--output",
        str(BASE / "all_features_v2.json"),
    )
    enc = run(
        "build_random_stream.py",
        "--features",
        str(BASE / "all_features_v2.json"),
    )
    compute = seg + feat + enc

    samples = recognition_times()
    median_rec = statistics.median(samples) if samples else 0.0
    mean_rec = statistics.mean(samples) if samples else 0.0

    print()
    print(f"crosses                : {crosses}")
    print(f"bits                   : {total_bits:,}")
    print(f"bits per cross         : {bits_per_cross:.1f}")
    print()
    print("stage timings (all crosses)")
    print(f"  segmentation         : {seg:6.2f} s")
    print(f"  feature extraction   : {feat:6.2f} s")
    print(f"  encoding             : {enc:6.2f} s")
    print(f"  compute subtotal     : {compute:6.2f} s")
    print(f"  Qwen recognition     : median {median_rec:.1f} s/crop, mean {mean_rec:.1f} s/crop")
    print()

    compute_rate = total_bits / compute
    end_to_end = compute + median_rec * crosses
    end_rate = total_bits / end_to_end

    print("throughput")
    print(
        f"  compute only         : {compute_rate:,.0f} bit/s  "
        f"= {compute_rate / 8 / 1e6:.6f} MB/s  "
        f"(= {compute_rate * 86400 / 8 / 1e9:.3f} GB/day)"
    )
    print(
        f"  end to end           : {end_rate:,.2f} bit/s  "
        f"= {end_rate / 8 / 1e6:.6f} MB/s  "
        f"(= {end_rate * 86400 / 8 / 1e9:.4f} GB/day)"
    )
    print(
        f"  per cross            : {compute / crosses * 1000:.1f} ms compute, "
        f"{median_rec:.1f} s recognition"
    )
    print()
    print(
        f"  one 1,000,000-bit stream takes about "
        f"{1_000_000 / compute_rate:.1f} s compute-only, "
        f"{1_000_000 / end_rate / 3600:.1f} h end-to-end"
    )

    (BASE / "final" / "throughput.json").write_text(
        json.dumps(
            {
                "crosses": crosses,
                "total_bits": total_bits,
                "bits_per_cross": bits_per_cross,
                "segmentation_s": seg,
                "features_s": feat,
                "encoding_s": enc,
                "compute_s": compute,
                "recognition_median_s": median_rec,
                "recognition_mean_s": mean_rec,
                "compute_bit_per_s": compute_rate,
                "compute_MB_per_s": compute_rate / 8 / 1e6,
                "end_to_end_bit_per_s": end_rate,
                "end_to_end_MB_per_s": end_rate / 8 / 1e6,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {BASE / 'final' / 'throughput.json'}")


if __name__ == "__main__":
    main()
