"""Check the effective degrees of freedom of similarity-fit residuals.

extract_cross_features() fits a similarity transform (scale, rotation, two
translation terms) from the 4 detected dots onto their canonical positions,
then stores the residuals as 8 numbers (dx, dy per dot). This script verifies
how many of those 8 numbers are actually free, using synthetic points so the
answer does not depend on the real images.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dot_features import similarity_transform


def residual_rows(points, target):
    scale, rotation, translation = similarity_transform(points, target)
    mapped = scale * (points @ rotation.T) + translation
    return (mapped - target).reshape(-1)


def main():
    target = np.array([[131.0, 31.0], [266.0, 33.0], [381.0, 153.0], [359.0, 257.0]])

    print("synthetic 4-dot similarity fits (8 residual numbers per fit)")
    for scale in (2.0, 6.0, 15.0):
        rng = np.random.default_rng(12345)
        rows = np.array(
            [
                residual_rows(target + rng.normal(scale=scale, size=target.shape), target)
                for _ in range(4000)
            ]
        )
        singular = np.linalg.svd(rows, compute_uv=False)
        nonzero = int((singular > singular.max() * 1e-6).sum())
        print(
            f"  noise sigma={scale:>4}: rank={nonzero}  singular={np.round(singular, 2).tolist()}"
        )

    print(
        "\nInterpretation: despite 8 residual columns, a 4-dot similarity fit "
        "leaves only about 5 independent directions. Listing dx/dy for all four "
        "dots therefore overstates the number of independent features."
    )


if __name__ == "__main__":
    main()
