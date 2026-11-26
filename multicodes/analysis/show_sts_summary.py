"""Print a readable per-test breakdown of the archived NIST STS runs."""

import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "nist_sts" / "runs"


def show(run_name):
    path = RUNS / run_name / "nist_summary.json"
    if not path.exists():
        print(f"missing: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    totals = data["totals"]
    print(f"=== {run_name} ===")
    print(f"    {data['streams']} stream(s) x {data['stream_length']} bits, alpha={data['alpha']}")
    print(
        f"    subtests={totals['subtests']}  passed={totals['passed']}  "
        f"failed={totals['failed']}  not_evaluated={totals['not_evaluated']}"
    )
    print(f"    {'test':<26}{'passed':>8}{'failed':>8}{'n/a':>6}")
    for t in data["tests"]:
        print(f"    {t['test']:<26}{t['passed']:>8}{t['failed']:>8}{t['not_evaluated']:>6}")
    print()

    failed = [(t["test"], d) for t in data["tests"] for d in t["details"] if d["pass"] is False]
    if failed:
        print("  failed subtests:")
        for name, d in failed:
            print(
                f"    {name:<26} #{d['index']:<4} "
                f"ratio={d['proportion']:<8} pooled_p={d['pooled_p_value']}"
            )
    skipped = [
        (t["test"], len([d for d in t["details"] if d["pass"] is None])) for t in data["tests"]
    ]
    skipped = [(n, c) for n, c in skipped if c]
    if skipped:
        print("  not evaluated:")
        for name, count in skipped:
            print(f"    {name:<26} {count} subtests")
    print()


if __name__ == "__main__":
    names = sys.argv[1:] or ["pca_v5_111", "pca_v5_222"]
    for name in names:
        show(name)
