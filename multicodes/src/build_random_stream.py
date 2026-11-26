"""Build the final bit stream from the combined feature table.

Per cross: concatenate the decimal digits of every feature (fixed feature
order), then base-convert the digit string to its binary expansion. Crosses are
concatenated into one stream with no hashing and no expansion.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from digit_encoding import digits_of, digits_to_bits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "qwen_recognition"
META = ("image", "group", "crop", "code", "confidence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(BASE / "all_features_v2.json"))
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="image groups to exclude, e.g. --exclude ccc",
    )
    parser.add_argument(
        "--output",
        default=str(BASE / "final" / "random_stream.bin"),
    )
    parser.add_argument(
        "--order",
        choices=["sequential", "interleave"],
        default="sequential",
        help=(
            "sequential: group by group; interleave: round-robin across "
            "groups. Order changes which crosses share a test block, not the "
            "measured feature data."
        ),
    )
    args = parser.parse_args()

    records = json.loads(Path(args.features).read_text(encoding="utf-8"))
    if args.groups:
        records = [r for r in records if r["group"] in args.groups]
    if args.exclude:
        excluded = set(args.exclude)
        records = [r for r in records if r["group"] not in excluded]
    if args.min_confidence > 0:
        records = [r for r in records if (r.get("confidence") or 0) >= args.min_confidence]

    names = [key for key in records[0] if key not in META]
    if args.order == "interleave":
        buckets = {}
        for record in records:
            buckets.setdefault(record["group"], []).append(record)
        ordered = []
        depth = 0
        while any(depth < len(v) for v in buckets.values()):
            for group in sorted(buckets):
                if depth < len(buckets[group]):
                    ordered.append(buckets[group][depth])
            depth += 1
        records = ordered
    parts = []
    for record in records:
        text = "".join("".join(str(d) for d in digits_of(record[name])) for name in names)
        parts.append(digits_to_bits(text))
    stream = "".join(parts)
    if not stream:
        raise SystemExit("empty stream")

    bits = np.frombuffer(stream.encode("ascii"), dtype=np.uint8) - ord("0")
    ones = int(bits.sum())
    ratio = ones / bits.size
    lag1 = float(np.corrcoef(bits[:-1], bits[1:])[0, 1]) if bits.size > 2 else 0.0
    print(f"crosses        : {len(records)}")
    print(f"features/cross : {len(names)}")
    print(f"bits/cross     : {stream.__len__() / len(records):.1f}")
    print(f"total bits     : {bits.size}")
    print(f"ones ratio     : {ratio:.4f}  (deviation {abs(ratio - 0.5):.4f})")
    print(f"lag-1          : {lag1:+.4f}")
    print(f"crosses for 1e6: {math.ceil(1_000_000 / (bits.size / len(records)))}")

    packed = bytes(
        int(stream[i : i + 8].ljust(8, "0"), 2) for i in range(0, len(stream) - len(stream) % 8, 8)
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(packed)
    (BASE / "final" / "stream_stats.json").write_text(
        json.dumps(
            {
                "crosses": len(records),
                "features_per_cross": len(names),
                "total_bits": int(bits.size),
                "bits_per_cross": bits.size / len(records),
                "ones_ratio": ratio,
                "lag1": lag1,
                "groups": {
                    g: sum(1 for r in records if r["group"] == g)
                    for g in sorted({r["group"] for r in records})
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"written to {out}")


if __name__ == "__main__":
    main()
