"""Convert measured feature values into decimal digits and binary strings."""

import math


LOG2_10 = math.log2(10)


def digits_of(value):
    """Return the decimal digits of a value, dropping sign and decimal point."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        text = f"{float(value):.17f}".rstrip("0")
    return [int(char) for char in text if char.isdigit()]


def digits_to_bits(text):
    """Map a decimal digit string to floor(len(text) * log2(10)) bits."""
    if not text:
        return ""
    count = int(len(text) * LOG2_10)
    value = int(text) << count
    return format(value // (10 ** len(text)), f"0{count}b")
