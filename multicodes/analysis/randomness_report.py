import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm


def bytes_to_bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def monobit_test(bits):
    signed = 2 * bits.astype(np.int64) - 1
    statistic = abs(int(signed.sum())) / math.sqrt(len(bits))
    p_value = erfc(statistic / math.sqrt(2.0))
    return {"p_value": float(p_value), "pass": bool(p_value >= 0.01)}


def block_frequency_test(bits, block_size=128):
    block_count = len(bits) // block_size
    if block_count == 0:
        return {"p_value": 0.0, "pass": False}
    trimmed = bits[: block_count * block_size].reshape(block_count, block_size)
    proportions = trimmed.mean(axis=1)
    chi_square = 4.0 * block_size * float(np.sum((proportions - 0.5) ** 2))
    p_value = gammaincc(block_count / 2.0, chi_square / 2.0)
    return {"p_value": float(p_value), "pass": bool(p_value >= 0.01)}


def runs_test(bits):
    proportion = float(bits.mean())
    if abs(proportion - 0.5) >= 2.0 / math.sqrt(len(bits)):
        return {"p_value": 0.0, "pass": False}
    transitions = int(np.count_nonzero(bits[1:] != bits[:-1]))
    runs = transitions + 1
    numerator = abs(runs - 2.0 * len(bits) * proportion * (1.0 - proportion))
    denominator = 2.0 * math.sqrt(2.0 * len(bits)) * proportion * (1.0 - proportion)
    p_value = erfc(numerator / denominator)
    return {"p_value": float(p_value), "pass": bool(p_value >= 0.01)}


def longest_run_test(bits, block_size=128):
    block_count = len(bits) // block_size
    if block_count == 0:
        return {"p_value": 0.0, "pass": False}
    classes = [0, 0, 0, 0, 0, 0]
    for block in bits[: block_count * block_size].reshape(block_count, block_size):
        longest = 0
        current = 0
        for value in block:
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        if longest <= 4:
            classes[0] += 1
        elif longest == 5:
            classes[1] += 1
        elif longest == 6:
            classes[2] += 1
        elif longest == 7:
            classes[3] += 1
        elif longest == 8:
            classes[4] += 1
        else:
            classes[5] += 1

    probabilities = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    chi_square = 0.0
    for observed, probability in zip(classes, probabilities):
        expected = block_count * probability
        chi_square += (observed - expected) ** 2 / expected
    p_value = gammaincc(3.0, chi_square / 2.0)
    return {"p_value": float(p_value), "pass": bool(p_value >= 0.01)}


def cumulative_sums_test(bits):
    def p_value_for(sequence):
        signed = 2 * sequence.astype(np.int64) - 1
        cumulative = np.cumsum(signed)
        z = int(np.max(cumulative))
        if z == 0:
            return 1.0
        root_n = math.sqrt(len(sequence))
        start = int((-len(sequence) / z + 1) / 4)
        end = int((len(sequence) / z - 1) / 4)
        first_sum = 0.0
        for k in range(start, end + 1):
            first_sum += norm.cdf((4 * k + 1) * z / root_n) - norm.cdf((4 * k - 1) * z / root_n)
        start = int((-len(sequence) / z - 3) / 4)
        end = int((len(sequence) / z - 1) / 4)
        second_sum = 0.0
        for k in range(start, end + 1):
            second_sum += norm.cdf((4 * k + 3) * z / root_n) - norm.cdf((4 * k + 1) * z / root_n)
        return float(max(0.0, min(1.0, 1.0 - first_sum + second_sum)))

    forward = p_value_for(bits)
    reverse = p_value_for(bits[::-1])
    return {
        "forward_p_value": forward,
        "reverse_p_value": reverse,
        "pass": bool(forward >= 0.01 and reverse >= 0.01),
    }


def spectral_test(bits):
    signed = 2 * bits.astype(np.float64) - 1
    spectrum = np.abs(np.fft.rfft(signed))
    threshold = math.sqrt(math.log(1.0 / 0.05) * len(bits))
    expected = 0.95 * len(bits) / 2.0
    observed = int(np.count_nonzero(spectrum < threshold))
    statistic = abs(observed - expected) / math.sqrt(len(bits) * 0.95 * 0.05 / 4.0)
    p_value = erfc(statistic / math.sqrt(2.0))
    return {"p_value": float(p_value), "pass": bool(p_value >= 0.01)}


def empirical_min_entropy(levels, level_count):
    counts = np.bincount(levels, minlength=level_count)
    probabilities = counts / counts.sum()
    p_hat = float(probabilities.max())
    confidence_bound = min(
        1.0,
        p_hat + 2.576 * math.sqrt(max(0.0, p_hat * (1.0 - p_hat)) / max(1, len(levels) - 1)),
    )
    return {
        "counts": counts.tolist(),
        "most_common_probability": p_hat,
        "min_entropy_point": -math.log2(p_hat),
        "min_entropy_mcv_99": -math.log2(confidence_bound),
    }


def entropy_assessment(root, group):
    features = json.loads((root / group / "features_pca.json").read_text(encoding="utf-8"))
    model = json.loads((root / "pca_model.json").read_text(encoding="utf-8"))
    component_count = int(model["components"])
    levels = []
    for record in features:
        bits = record["feature_bits"]
        levels.append([int(bits[index : index + 2], 2) for index in range(0, len(bits), 2)])
    levels = np.array(levels)

    per_component = []
    total_point = 0.0
    total_mcv_99 = 0.0
    for component in range(component_count):
        estimate = empirical_min_entropy(levels[:, component], 4)
        estimate["component"] = component + 1
        per_component.append(estimate)
        total_point += estimate["min_entropy_point"]
        total_mcv_99 += estimate["min_entropy_mcv_99"]

    return {
        "crosses": len(features),
        "components": component_count,
        "total_component_entropy_point_estimate": total_point,
        "total_component_entropy_mcv_99": total_mcv_99,
        "per_component": per_component,
        "warning": (
            "These are empirical estimates from a very small sample. "
            "They are not NIST SP 800-90B certification results."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    report = {}

    for group in args.groups:
        data = (root / group / "expanded_1000000.bin").read_bytes()
        bits = bytes_to_bits(data)
        tests = {
            "monobit": monobit_test(bits),
            "block_frequency": block_frequency_test(bits),
            "runs": runs_test(bits),
            "longest_run": longest_run_test(bits),
            "cumulative_sums": cumulative_sums_test(bits),
            "spectral": spectral_test(bits),
        }
        report[group] = {
            "bits": len(bits),
            "tests": tests,
            "entropy_assessment": entropy_assessment(root, group),
        }

    (root / "randomness_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
