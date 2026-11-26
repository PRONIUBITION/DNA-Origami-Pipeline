"""Explain why dumping float64 features produces so many zero bits.

For every feature value this reports:
  * whether the value is an exact integer
  * how many trailing zero bits its float64 mantissa has
  * the actual 64-bit pattern of a few examples
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np


def bits_of(value):
    (raw,) = struct.unpack(">Q", struct.pack(">d", float(value)))
    return format(raw, "064b")


def trailing_zeros(binary):
    return len(binary) - len(binary.rstrip("0"))


def main(paths):
    values = []
    for path in paths:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        for record in records:
            values.extend(record.get("all_features") or record["features"])
    values = np.array(values, float)

    print(f"feature values analysed : {values.size}")
    print(f"distinct values         : {len(set(values.tolist()))}")
    print()

    integers = values == np.floor(values)
    print(f"exact integers          : {int(integers.sum())} ({integers.mean() * 100:.1f}%)")
    near = np.abs(values - np.round(values)) < 1e-9
    print(f"within 1e-9 of integer  : {int(near.sum())} ({near.mean() * 100:.1f}%)")
    print(
        f"values with 1 decimal   : "
        f"{int((np.abs(values * 10 - np.round(values * 10)) < 1e-9).sum())}"
    )
    print()

    patterns = [bits_of(v) for v in values]
    zeros = np.array([trailing_zeros(p) for p in patterns])
    ones = np.array([p.count("1") for p in patterns])

    print(f"ones per 64-bit value   : mean {ones.mean():.1f} / 64 ({ones.mean() / 64 * 100:.1f}%)")
    print(
        f"trailing zero bits      : mean {zeros.mean():.1f}, "
        f"median {int(np.median(zeros))}, max {zeros.max()}"
    )
    print()

    total_ones = int(ones.sum())
    total_bits = values.size * 64
    print(f"overall ones ratio      : {total_ones / total_bits:.4f}")
    print()

    print("examples:")
    for v in values[:6]:
        print(f"  {v:>12.6f}  ->  {bits_of(v)}")
    print()

    # Where do the ones actually live?
    sign = np.array([int(p[0]) for p in patterns])
    exponent = np.array([p[1:12] for p in patterns])
    mantissa = np.array([p[12:] for p in patterns])
    print(f"sign bit ones           : {sign.mean() * 100:.1f}%")
    print(
        f"exponent bits ones      : "
        f"{sum(b.count('1') for b in exponent) / (values.size * 11) * 100:.1f}%"
    )
    print(
        f"mantissa bits ones      : "
        f"{sum(b.count('1') for b in mantissa) / (values.size * 52) * 100:.1f}%"
    )
    print(f"distinct exponents      : {len(set(exponent.tolist()))}")


if __name__ == "__main__":
    main(sys.argv[1:])
