import numpy as np
import cv2

NN_THRESH = 0.7


def match(desc1, desc2, nn_thresh=NN_THRESH):
    """
    Mutual nearest-neighbour matching between two descriptor sets.

    Args:
        desc1 : (256, N1)
        desc2 : (256, N2)

    Returns:
        matches : list of (idx1, idx2) pairs
    """
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return []

    # L2 normalize
    d1 = desc1 / (np.linalg.norm(desc1, axis=0, keepdims=True) + 1e-8)
    d2 = desc2 / (np.linalg.norm(desc2, axis=0, keepdims=True) + 1e-8)

    dmat = np.dot(d1.T, d2)           # (N1, N2) similarity matrix
    idx1 = np.argmax(dmat, axis=1)    # best match in desc2 for each in desc1
    idx2 = np.argmax(dmat, axis=0)    # best match in desc1 for each in desc2

    matches = []
    for i, j in enumerate(idx1):
        if idx2[j] == i and dmat[i, j] >= nn_thresh:
            matches.append((i, j))

    return matches


def compute_homography(pts_template, pts_test, matches):
    """
    RANSAC homography from matched keypoint pairs.

    Args:
        pts_template : (3, N)  template keypoints (x, y, conf)
        pts_test     : (3, N)  test keypoints
        matches      : list of (idx_template, idx_test)

    Returns:
        H    : (3, 3) homography matrix, None if failed
        mask : inlier boolean array, None if failed
    """
    if len(matches) < 4:
        print(f"Not enough matches to compute homography: {len(matches)}")
        return None, None

    src = np.array([[pts_template[0, i], pts_template[1, i]] for i, _ in matches], dtype=np.float32)
    dst = np.array([[pts_test[0, j],     pts_test[1, j]]     for _, j in matches], dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H, mask