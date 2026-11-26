"""Analyze the confidence level of distance groupings
Check the ratio of the smallest distance to the second smallest distance"""

import numpy as np
from pathlib import Path
import json
from itertools import combinations


def analyze_distance_confidence(keypoints):
    """Analyze the confidence level of distance groupings
    Return: smallest distance, second smallest distance, confidence (ratio)"""
    valid_indices = np.where(keypoints[:, 2] > 0)[0]
    valid_points = keypoints[valid_indices][:, :2]

    if len(valid_points) != 4:
        return None

    # Calculate all 6 pairs of distances
    pairs = list(combinations(range(4), 2))
    pair_dists = []
    for i, j in pairs:
        dist = np.linalg.norm(valid_points[i] - valid_points[j])
        pair_dists.append((dist, i, j))

    # Sorting
    pair_dists.sort()

    min_dist = pair_dists[0][0]
    second_min_dist = pair_dists[1][0]
    third_min_dist = pair_dists[2][0]

    # Confidence: Second Smallest Distance / Smallest Distance
    # The higher the ratio, the more significant the minimum distance and the more reliable the grouping.
    confidence_ratio = second_min_dist / (min_dist + 1e-6)

    return {
        "min_dist": min_dist,
        "second_min_dist": second_min_dist,
        "third_min_dist": third_min_dist,
        "confidence_ratio": confidence_ratio,
        "pairs": pair_dists,
    }


def main():
    # Load comparison results
    result_file = (
        Path(__file__).resolve().parents[1]
        / "stage2_hrnet"
        / "outputs"
        / "grouping_comparison_v2"
        / "comparison_results.json"
    )
    if not result_file.exists():
        print("""Result file not found""")
        return

    with open(result_file, "r") as f:
        results = json.load(f)

    # Only 4-point samples with inconsistent classifications are analyzed
    four_dot_results = [r for r in results if r["num_dots"] == 4]
    diff_results = [r for r in four_dot_results if not r["same"]]

    print(f"""Total 4-point samples: {len(four_dot_results)}""")
    print(f"""Inconsistent classification: {len(diff_results)}""")
    print()

    # Analysis of Confidence Distribution
    low_confidence = []  # Ratio < 1.5 (may be ambiguous)
    medium_confidence = []  # 1.5 <= Ratio < 2.0
    high_confidence = []  # Ratio >= 2.0

    for result in four_dot_results:
        # Load image from file name to obtain keypoints
        # Simplified: Suppose we have the original keypoints data.
        # Here, we use the distance information in "distances" for estimation.

        pca_dist1 = result["pca"]["side1_dist"]
        pca_dist2 = result["pca"]["side2_dist"]
        dist_dist1 = result["distance"]["side1_dist"]
        dist_dist2 = result["distance"]["side2_dist"]

        # Estimate confidence (using the difference in distances between the PCA and distance methods)
        # If the distances calculated by the two methods differ significantly, it indicates ambiguity.
        avg_pca = (pca_dist1 + pca_dist2) / 2
        avg_dist = (dist_dist1 + dist_dist2) / 2

        if avg_dist > 0:
            diff_ratio = abs(avg_pca - avg_dist) / avg_dist
        else:
            diff_ratio = 0

        if result["same"]:
            if diff_ratio < 0.2:
                high_confidence.append(result)
            else:
                medium_confidence.append(result)
        else:
            if diff_ratio < 0.5:
                low_confidence.append(result)
            else:
                medium_confidence.append(result)

    print("""Confidence analysis (based on the distance difference between the two methods):""")
    print(f"""High confidence (consistent): {len(high_confidence)}""")
    print(f"""Moderate confidence: {len(medium_confidence)}""")
    print(f"""Low confidence (inconsistent): {len(low_confidence)}""")
    print()

    # Examine the characteristics of distance-based groupings
    print("""Distance difference between distance-based grouping and PCA-based grouping:""")
    same_dists = []
    diff_dists = []

    for r in four_dot_results:
        pca_d1 = r["pca"]["side1_dist"]
        pca_d2 = r["pca"]["side2_dist"]
        dist_d1 = r["distance"]["side1_dist"]
        dist_d2 = r["distance"]["side2_dist"]

        # Calculate the distance difference between the two methods
        diff1 = abs(pca_d1 - dist_d1)
        diff2 = abs(pca_d2 - dist_d2)
        avg_diff = (diff1 + diff2) / 2

        if r["same"]:
            same_dists.append(avg_diff)
        else:
            diff_dists.append(avg_diff)

    if same_dists:
        print(
            f"""Average distance difference for samples with consistent classification: {np.mean(same_dists):.1f} px"""
        )
    if diff_dists:
        print(
            f"""Average distance difference for samples with inconsistent classifications: {np.mean(diff_dists):.1f} px"""
        )

    print()
    print("""Conclusion:""")
    if diff_dists and same_dists:
        if np.mean(diff_dists) > np.mean(same_dists) * 2:
            print(
                """Samples with inconsistent classifications do indeed have significantly larger distance differences."""
            )
            print(
                """This indicates that distance-based grouping is more reliable in these cases."""
            )
        else:
            print(
                """! The difference between the two methods is not significant enough and requires a more careful examination."""
            )


if __name__ == "__main__":
    main()
