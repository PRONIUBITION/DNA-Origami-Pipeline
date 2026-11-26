"""Map a chain stem to its single-cross template stem.

111..999 pair with 1..9; aaa..yyy pair with a..y.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def template_stem(chain_stem):
    stem = chain_stem.lower()
    if len(stem) == 3 and stem[0] == stem[1] == stem[2]:
        if stem[0].isdigit() or stem[0].isalpha():
            return stem[0]
    raise ValueError(f"cannot derive template for {chain_stem}")


def template_path(chain_stem):
    return ROOT / "input" / "single" / f"{template_stem(chain_stem)}.png"


def chain_path(chain_stem):
    return ROOT / "input" / "multi" / f"{chain_stem}.png"
