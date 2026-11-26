"""Inspect the packed bit stream produced by build_random_stream.py.

The .bin stores 8 bits per byte, most significant bit first, so byte 0x92 is
the bit sequence 10010010.
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/qwen_recognition/final/random_stream.bin"


def load(path):
    data = np.fromfile(path, dtype=np.uint8)
    return data, np.unpackbits(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(DEFAULT))
    parser.add_argument("--head", type=int, default=0, help="print the first N bits")
    parser.add_argument("--tail", type=int, default=0, help="print the last N bits")
    parser.add_argument("--offset", type=int, default=0, help="start bit for --count")
    parser.add_argument("--count", type=int, default=0, help="print N bits starting at --offset")
    parser.add_argument("--hex", type=int, default=0, help="print the first N bytes in hex")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--to-txt", default=None, help="write the whole stream as ASCII 0/1")
    parser.add_argument("--to-hex", default=None, help="write the whole stream as hex text")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    data, bits = load(path)

    print(f"file      : {path}")
    print(f"bytes     : {len(data):,}")
    print(f"bits      : {bits.size:,}")

    if args.stats:
        ones = int(bits.sum())
        print(f"ones      : {ones:,} ({ones / bits.size:.6f})")
        print(f"zeros     : {bits.size - ones:,} ({(bits.size - ones) / bits.size:.6f})")
        transitions = int((bits[1:] != bits[:-1]).sum())
        print(f"transitions: {transitions:,} ({transitions / (bits.size - 1):.6f})")

    if args.hex:
        shown = data[: args.hex]
        print(f"\nfirst {len(shown)} bytes:")
        for start in range(0, len(shown), 16):
            chunk = shown[start : start + 16]
            text = " ".join(f"{b:02X}" for b in chunk)
            print(f"  {start:08X}  {text}")

    def show(label, values):
        if values.size == 0:
            return
        print(f"\n{label} ({values.size} bits):")
        text = "".join(str(int(v)) for v in values)
        for start in range(0, len(text), 64):
            line = text[start : start + 64]
            print(f"  {start:>8}  {' '.join(line[i : i + 8] for i in range(0, len(line), 8))}")

    if args.head:
        show("head", bits[: args.head])
    if args.tail:
        show("tail", bits[-args.tail :])
    if args.count:
        show(f"offset {args.offset}", bits[args.offset : args.offset + args.count])

    if args.to_txt:
        out = Path(args.to_txt)
        out.write_text("".join(str(int(v)) for v in bits), encoding="ascii")
        print(f"\nASCII 0/1 written to {out} ({bits.size:,} characters)")
    if args.to_hex:
        out = Path(args.to_hex)
        out.write_text(data.tobytes().hex(), encoding="ascii")
        print(f"\nhex written to {out} ({len(data) * 2:,} characters)")


if __name__ == "__main__":
    main()
