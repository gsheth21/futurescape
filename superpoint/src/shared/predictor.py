import numpy as np


def project_keypoints(canonical_kps, H):
    """
    Warp canonical template keypoints into new image space using homography H.

    Args:
        canonical_kps : (N, 2)  x,y coords defined on the template
        H             : (3, 3)  homography  (template -> new image)

    Returns:
        projected : (N, 2)  x,y coords in new image space
    """
    if H is None or len(canonical_kps) == 0:
        return np.empty((0, 2))

    ones      = np.ones((len(canonical_kps), 1), dtype=np.float32)
    kps_h     = np.hstack([canonical_kps, ones])     # (N, 3)
    proj_h    = (H @ kps_h.T).T                       # (N, 3)
    projected = proj_h[:, :2] / proj_h[:, 2:3]        # dehomogenize
    return projected
