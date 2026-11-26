import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "multicodes" / "src"
sys.path.insert(0, str(SRC))

from digit_encoding import digits_of, digits_to_bits  # noqa: E402


def test_decimal_digit_encoding():
    assert digits_of(-5.713803592761764) == [5, 7, 1, 3, 8, 0, 3, 5, 9, 2, 7, 6, 1, 7, 6, 4]
    assert digits_of(151.0) == [1, 5, 1, 0]
    assert digits_of(0.1199132376163945) == [
        0,
        1,
        1,
        9,
        9,
        1,
        3,
        2,
        3,
        7,
        6,
        1,
        6,
        3,
        9,
        4,
        5,
    ]
    assert len(digits_to_bits("123456789")) == 29


def test_no_hash_expansion_code_in_multicodes():
    forbidden = (
        "import hashlib",
        "sha256(",
        "expand_bits(",
        "cross_hash",
        "whitedot_random_expand",
    )
    offenders = []
    for path in (ROOT / "multicodes").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in forbidden):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_obsolete_experimental_modules_removed():
    removed = (
        "whitedot_random_expand.py",
        "whitedot_random_expand_pca.py",
        "seed_quality_eval.py",
        "dataset_size_estimator.py",
        "raw_feature_bits.py",
        "bcd_encoding.py",
        "encode_features_v2.py",
        "search_partition_dft.py",
        "search_segmentation.py",
    )
    for filename in removed:
        assert not list((ROOT / "multicodes").rglob(filename))
