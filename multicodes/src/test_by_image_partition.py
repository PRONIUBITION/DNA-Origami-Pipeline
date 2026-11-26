"""Test three blocks defined by disjoint sets of AFM source images.

Each block uses its own sequence length: floor(block_bits / 56). Rank is run
separately on the full block stream, because 56 short sequences cannot build
enough 32x32 matrices for a meaningful rank p-value distribution.

Also copies the source chain images and their single-cross templates into
outputs/qwen_recognition/final/image_partition/partN/.
"""

import json
import shutil
import subprocess
import argparse
from pathlib import Path

import numpy as np

from digit_encoding import digits_of, digits_to_bits
from pairing import template_path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
BASE = ROOT / "outputs/qwen_recognition"
PART_JSON = BASE / "final/image_partition_by_crosses.json"
WORK = BASE / "final/image_partition"
RUNS = ROOT / "nist_sts/runs"
PYTHON = __import__("sys").executable
META = ("image", "group", "crop", "code", "confidence")
STREAMS = 56

PANEL = [
    ("Frequency", "frequency"),
    ("BlockFrequency", "block frequency"),
    ("CumulativeSums", "cumulative sums"),
    ("Runs", "runs"),
    ("LongestRun", "longest run"),
    ("Rank", "rank"),
    ("FFT", "discrete Fourier transform"),
    ("ApproximateEntropy", "approximate entropy"),
    ("Serial", "serial"),
]


def bits_for(record, names):
    text = "".join("".join(str(d) for d in digits_of(record[name])) for name in names)
    return digits_to_bits(text)


def write_images(part, images, work):
    target = work / f"part{part}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for stem in images:
        chain_src = ROOT / "input/multi" / f"{stem}.png"
        tmpl_src = template_path(stem)
        if chain_src.exists():
            shutil.copy2(chain_src, target / chain_src.name)
        if tmpl_src.exists():
            shutil.copy2(tmpl_src, target / f"{stem}_template_{tmpl_src.name}")
        manifest.append(
            {
                "chain": chain_src.name,
                "template": tmpl_src.name,
                "template_copied_as": f"{stem}_template_{tmpl_src.name}",
            }
        )
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def run_sts(binary, name, length, streams, extra, prefix):
    out = RUNS / f"{prefix}_{name}"
    cmd = [
        PYTHON,
        str(SRC / "run_official_sts.py"),
        "--input",
        str(binary),
        "--name",
        name,
        "--length",
        str(length),
        "--streams",
        str(streams),
        "--output-dir",
        str(out),
        *extra,
    ]
    subprocess.run(
        cmd, cwd=str(ROOT), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return json.loads((out / "nist_summary.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default=str(PART_JSON))
    parser.add_argument("--work", default=str(WORK))
    parser.add_argument("--prefix", default="imgpart")
    args = parser.parse_args()

    part_json = Path(args.partition)
    work = Path(args.work)
    prefix = args.prefix

    partition = json.loads(part_json.read_text(encoding="utf-8"))
    records = json.loads((BASE / "all_features_v2.json").read_text(encoding="utf-8"))
    names = [k for k in records[0] if k not in META]
    by_group = {}
    for record in records:
        by_group.setdefault(record["group"], []).append(record)

    work.mkdir(parents=True, exist_ok=True)
    results = []
    for part in partition["parts"]:
        index = part["part"]
        images = part["images"]
        write_images(index, images, work)

        chunks = []
        for stem in images:
            for record in by_group.get(stem, []):
                chunks.append(bits_for(record, names))
        stream = "".join(chunks)
        length = len(stream) // STREAMS
        used = length * STREAMS

        full = work / f"part{index}_full.bin"
        test = work / f"part{index}.bin"
        full.write_bytes(
            np.packbits(np.frombuffer(stream.encode("ascii"), dtype=np.uint8) - ord("0")).tobytes()
        )
        test.write_bytes(
            np.packbits(
                np.frombuffer(stream[:used].encode("ascii"), dtype=np.uint8) - ord("0")
            ).tobytes()
        )

        summary = run_sts(
            test,
            f"part{index}_{STREAMS}x{length}",
            length,
            STREAMS,
            ["--apen-m", "4", "--serial-m", "8", "--p-threshold", "0.001"],
            prefix,
        )
        rank = run_sts(full, f"part{index}_rank_single", len(stream), 1, [], prefix)
        rank_detail = {t["test"]: t for t in rank["tests"]}["Rank"]["details"][0]

        by_name = {t["test"]: t for t in summary["tests"]}
        failed = [
            key for key, _ in PANEL if by_name[key]["failed"] > 0 or by_name[key]["subtests"] == 0
        ]
        results.append(
            {
                "part": index,
                "images": images,
                "crosses": sum(len(by_group.get(s, [])) for s in images),
                "bits": len(stream),
                "sequence_length": length,
                "used_bits": used,
                "discarded_bits": len(stream) - used,
                "failed": failed,
                "by_name": by_name,
                "rank_single": rank_detail,
            }
        )
        print(
            f"part {index}: {len(images)} images, {len(stream):,} bits, "
            f"{STREAMS} x {length:,} = {used:,} "
            f"(discard {len(stream) - used})  "
            f"{9 - len(failed)}/9 pass" + (f"  failing {failed}" if failed else "")
        )

    print()
    for item in results:
        print(f"===== part {item['part']} =====")
        for key, label in PANEL:
            if key == "Rank":
                detail = item["rank_single"]
                print(
                    f"  {label:<24}{detail['proportion']:>8}  "
                    f"p={detail['pooled_p_value']}  "
                    f"{'PASS' if detail['pass'] else 'FAIL'}"
                )
                continue
            detail = item["by_name"][key]["details"][0]
            print(
                f"  {label:<24}{detail['proportion']:>8}  "
                f"p={detail['pooled_p_value']:<10}  "
                f"{'PASS' if detail['pass'] else 'FAIL'}"
            )
        print()

    (work / "results.json").write_text(
        json.dumps(
            [
                {
                    "part": r["part"],
                    "images": r["images"],
                    "crosses": r["crosses"],
                    "bits": r["bits"],
                    "sequence_length": r["sequence_length"],
                    "used_bits": r["used_bits"],
                    "discarded_bits": r["discarded_bits"],
                    "failed": r["failed"],
                    "rank_single": r["rank_single"],
                    "tests": {
                        key: r["by_name"][key]["details"][0] for key, _ in PANEL if key != "Rank"
                    },
                }
                for r in results
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"written to {work / 'results.json'}")


if __name__ == "__main__":
    main()
