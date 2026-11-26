"""Draw the two NIST summary charts used in the reference document.

  1. pass rates per test, with the expected rate (1 - alpha) as a dashed line
     and the lower 3-sigma acceptance bound as a solid line
  2. uniformity p-value per test, with the 0.0001 threshold marked

Reads an archived nist_summary.json produced by run_official_sts.py.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

BLUE = "#1f77b4"
RED = "#c0392b"
GREY = "#555555"


def pretty(name):
    table = {
        "Frequency": "Frequency",
        "BlockFrequency": "Block frequency",
        "CumulativeSums": "Cumulative sums",
        "Runs": "Runs",
        "LongestRun": "Longest run",
        "Rank": "Rank",
        "FFT": "DFT",
        "NonOverlappingTemplate": "Non-overlap template",
        "OverlappingTemplate": "Overlap template",
        "Universal": "Universal",
        "ApproximateEntropy": "Approximate entropy",
        "RandomExcursions": "Random excursions",
        "RandomExcursionsVariant": "Random excursions variant",
        "Serial": "Serial",
        "LinearComplexity": "Linear complexity",
    }
    return table.get(name, name)


def set_xticklabels(ax, labels, fontsize=8, rotation=45):
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(
        labels,
        fontsize=fontsize,
        rotation=rotation,
        ha="right" if rotation else "center",
        rotation_mode="anchor",
    )


PANEL = [
    "Frequency",
    "BlockFrequency",
    "CumulativeSums",
    "Runs",
    "LongestRun",
    "Rank",
    "FFT",
    "ApproximateEntropy",
    "Serial",
]


def collect(summary, keep, all_subtests=False):
    labels, rates, uniformity = [], [], []
    for test in summary["tests"]:
        if keep and test["test"] not in keep:
            continue
        details = test["details"]
        selected = list(enumerate(details, start=1))
        if not all_subtests:
            selected = selected[:1]
        multi = len(details) > 1 and all_subtests
        for index, item in selected:
            if not item.get("proportion") or "/" not in item["proportion"]:
                continue
            passed, total = (int(v) for v in item["proportion"].split("/"))
            if total == 0:
                continue
            label = pretty(test["test"])
            if multi:
                label = f"{label}\n#{index}"
            labels.append(label)
            rates.append(passed / total)
            uniformity.append(item.get("pooled_p_value"))
    return labels, np.array(rates), uniformity


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(width=1.0, labelsize=9)
    ax.grid(False)


def save_figure(fig, out_base):
    """Save PNG plus two SVGs: editable text and outlined text.

    PowerPoint sometimes clips rotated text when SVG text stays editable, so
    the outlined copy uses paths and cannot be cropped by a text frame.
    """
    fig.savefig(f"{out_base}.png", dpi=300)
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig.savefig(f"{out_base}.svg")
    matplotlib.rcParams["svg.fonttype"] = "path"
    fig.savefig(f"{out_base}_outlined.svg")
    matplotlib.rcParams["svg.fonttype"] = "none"


def plot_pass_rates(labels, rates, alpha, sample_size, out_base, title, rotation=45):
    expected = 1.0 - alpha
    # The STS prints the acceptance threshold as an integer count
    # (approximately mean - 3 sigma, truncated); use that count directly.
    lower_count = int((expected - 3.0 * math.sqrt(alpha * expected / sample_size)) * sample_size)
    lower = lower_count / sample_size

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    x = np.arange(len(labels))
    # Draw the bars from the visible axis floor instead of from zero. The
    # rectangles then lie entirely inside the axes, so PowerPoint keeps them
    # correct even after ungrouping drops the SVG clip path.
    baseline = min(lower, float(rates.min())) - 0.03
    ax.bar(
        x,
        rates - baseline,
        bottom=baseline,
        width=0.62,
        color=BLUE,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.axhline(lower, color=GREY, linestyle="-", linewidth=1.2, zorder=4)
    ax.set_ylim(baseline, 1.005)
    set_xticklabels(ax, labels, 8, rotation)
    ax.set_ylabel("Pass rate", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    style_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    save_figure(fig, out_base)
    plt.close(fig)
    return expected, lower


def plot_uniformity(labels, uniformity, out_base, title, threshold=1e-4, rotation=45):
    values = np.array([np.nan if v is None else v for v in uniformity], float)
    floor = threshold / 10.0
    plot_values = np.where(np.isnan(values) | (values <= 0), floor, np.maximum(values, floor))

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    x = np.arange(len(labels))
    # Same clip-path fix as above: bars start at the visible log floor.
    baseline = floor / 1.5
    ax.bar(
        x,
        plot_values - baseline,
        bottom=baseline,
        width=0.62,
        color=BLUE,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylim(baseline, 1.2)
    ax.axhline(threshold, color=RED, linestyle="--", linewidth=1.4, zorder=4)
    set_xticklabels(ax, labels, 8, rotation)
    ax.set_ylabel("Uniformity p-value (log scale)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    style_axes(ax)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v <= floor else f"{v:g}"))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    save_figure(fig, out_base)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default=str(Path(__file__).resolve().parents[1] / "nist_sts/runs/strict_m4_s8"),
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "outputs/qwen_recognition/final/charts"),
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--prefix", default="nist")
    parser.add_argument(
        "--tests",
        nargs="*",
        default=PANEL,
        help="test names to plot; pass an empty list to plot all",
    )
    parser.add_argument(
        "--rank-p",
        type=float,
        default=None,
        help=(
            "replace the rank bar with a single-stream result: pass rate 1.0 "
            "and this p-value (for protocols that test rank separately)"
        ),
    )
    parser.add_argument(
        "--all-subtests",
        action="store_true",
        help="plot every subtest (e.g. cumulative sums #2); default is one bar per test",
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="plot all 15 tests instead of the nine-test panel",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=45,
        help="x tick label rotation in degrees (0 for horizontal)",
    )
    args = parser.parse_args()

    run = Path(args.run)
    summary = json.loads((run / "nist_summary.json").read_text(encoding="utf-8"))
    keep = None if args.all_tests else (set(args.tests) if args.tests else None)
    labels, rates, uniformity = collect(summary, keep, args.all_subtests)
    if args.rank_p is not None:
        for index, label in enumerate(labels):
            if label.startswith("Rank"):
                rates[index] = 1.0
                uniformity[index] = args.rank_p
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sample_size = summary["streams"]
    expected, lower = plot_pass_rates(
        labels,
        rates,
        args.alpha,
        sample_size,
        out / f"{args.prefix}_pass_rates",
        "Pass rates for statistical tests of the NIST STS",
        args.rotate,
    )
    plot_uniformity(
        labels,
        uniformity,
        out / f"{args.prefix}_uniformity",
        "Uniformity of p-values for histograms of tests",
        rotation=args.rotate,
    )

    print(f"sequences       : {sample_size}")
    print(f"stream length   : {summary['stream_length']}")
    print(f"expected rate   : {expected:.4f}")
    print(f"3-sigma bound   : {lower:.4f}")
    print(f"minimum rate    : {rates.min():.4f}")
    valid = [v for v in uniformity if v is not None]
    print(f"min uniformity  : {min(valid):.6f}" if valid else "")
    print(f"charts written to {out}")


if __name__ == "__main__":
    main()
