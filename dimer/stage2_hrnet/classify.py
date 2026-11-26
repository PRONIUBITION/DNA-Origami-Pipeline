"""DNA dimer classification script
Classification based on the positions of the small white dots detected by HRNet

Classification criteria:
- Calculate the distance between the two left points and the distance between the two right points
- Distance < threshold (default 18px): Type 1 (small distance)
- Distance >= threshold (default 18px): Type 0 (large distance)
- Combinations obtained: 00, 01, 10, 11"""

import numpy as np


def calc_distance(p1, p2):
    """Calculate the Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def classify_dimer(keypoints, threshold=18.0):
    """Classify DNA dimers based on 4 key points

    Args:
    keypoints: numpy array [4, 3] -> [[x, y, visibility], ...]
    Order: left_dot_1, left_dot_2, right_dot_1, right_dot_2
    threshold: Distance threshold; less than is Type 1, greater than or equal to is Type 0

    Returns:
    class_name: str -> "00", "01", "10", "11"
    details: dict -> Dictionary containing detailed information"""
    # Extract coordinates
    left_1 = keypoints[0, :2]  # left_dot_1
    left_2 = keypoints[1, :2]  # left_dot_2
    right_1 = keypoints[2, :2]  # right_dot_1
    right_2 = keypoints[3, :2]  # right_dot_2

    # Check visibility
    left_visible = keypoints[0, 2] > 0 and keypoints[1, 2] > 0
    right_visible = keypoints[2, 2] > 0 and keypoints[3, 2] > 0

    details = {
        "left_distance": None,
        "right_distance": None,
        "left_type": None,
        "right_type": None,
        "left_visible": left_visible,
        "right_visible": right_visible,
    }

    # Calculate left type
    if left_visible:
        left_dist = calc_distance(left_1, left_2)
        left_type = 1 if left_dist < threshold else 0
        details["left_distance"] = left_dist
        details["left_type"] = left_type
    else:
        left_type = None

    # Calculate right type
    if right_visible:
        right_dist = calc_distance(right_1, right_2)
        right_type = 1 if right_dist < threshold else 0
        details["right_distance"] = right_dist
        details["right_type"] = right_type
    else:
        right_type = None

    # Combination category
    if left_type is not None and right_type is not None:
        class_name = f"{left_type}{right_type}"
    else:
        class_name = "incomplete"  # Missing points

    return class_name, details


def batch_classify(keypoints_batch, threshold=18.0):
    """Batch Classification

    Args:
    keypoints_batch: list of NumPy arrays [4, 3]
    threshold: Distance threshold

    Returns:
    classes: list of str
    all_details: list of dict"""
    classes = []
    all_details = []

    for keypoints in keypoints_batch:
        class_name, details = classify_dimer(keypoints, threshold)
        classes.append(class_name)
        all_details.append(details)

    return classes, all_details


# Class name mapping (for visualization)
CLASS_NAMES = {
    "00": "Type 00 (Both Large)",
    "01": "Type 01 (Left Large, Right Small)",
    "10": "Type 10 (Left Small, Right Large)",
    "11": "Type 11 (Both Small)",
    "incomplete": "Incomplete (Missing Points)",
}


if __name__ == "__main__":
    # Test example
    # Type 11: Small distance on both sides
    kp_11 = np.array(
        [
            [50, 50, 2],  # left_dot_1
            [55, 55, 2],  # left_dot_2 (distance ~7px, small)
            [100, 50, 2],  # right_dot_1
            [105, 52, 2],  # right_dot_2 (distance ~4px, small)
        ]
    )

    # Type 00: Large spacing on both sides
    kp_00 = np.array(
        [
            [50, 50, 2],  # left_dot_1
            [70, 60, 2],  # left_dot_2 (distance ~22px, large)
            [100, 50, 2],  # right_dot_1
            [125, 55, 2],  # right_dot_2 (distance ~26px, large)
        ]
    )

    print("""Test category:""")
    for name, kp in [
        ("""11 (should be a small distance)""", kp_11),
        ("""00 (should be a long distance)""", kp_00),
    ]:
        cls, det = classify_dimer(kp)
        print(f"\n{name}:")
        print(f"""Classification result: {cls}""")
        print(f"""Left distance: {det["left_distance"]:.2f} px, Type: {det["left_type"]}""")
        print(f"""Right distance: {det["right_distance"]:.2f} px, Type: {det["right_type"]}""")
