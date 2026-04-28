"""
SuperPoint-based Board Game Matcher
Uses MagicLeap's pretrained SuperPoint network.
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / '.env')

# Add SuperPoint to path
sys.path.append(str(Path(__file__).parent / '../../SuperPointPretrainedNetwork'))
from demo_superpoint import SuperPointFrontend

_SP_WEIGHTS = os.getenv('SP_WEIGHTS', str(Path(__file__).parents[2] / 'SuperPointPretrainedNetwork/pretrained/superpoint_v1.pth'))


class SuperPointBoardMatcher:
    def __init__(self, weights_path=None,
                 nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=True):
        """
        Initialize SuperPoint matcher.

        Args:
            weights_path: Path to pretrained weights (defaults to SP_WEIGHTS env var)
            nms_dist: Non-maximum suppression distance
            conf_thresh: Confidence threshold for keypoints
            nn_thresh: Nearest neighbor matching threshold
            cuda: Use GPU if available
        """
        if weights_path is None:
            weights_path = _SP_WEIGHTS

        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh

        # Check CUDA availability
        self.cuda = cuda and torch.cuda.is_available()
        if self.cuda:
            print("Using GPU for SuperPoint")
        else:
            print("Using CPU for SuperPoint")

        # Initialize SuperPoint
        self.superpoint = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=self.cuda
        )

        print(f"SuperPoint initialized (NMS={nms_dist}, conf={conf_thresh})")

    def preprocess_image(self, image):
        """
        Preprocess image for SuperPoint.

        Args:
            image: BGR image from OpenCV

        Returns:
            Grayscale float32 image normalized to [0, 1]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = gray.astype('float32') / 255.0
        return gray

    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors using SuperPoint.

        Args:
            image: BGR image

        Returns:
            pts: Keypoint coordinates (Nx2)
            desc: Descriptors (256xN)
            scores: Keypoint confidence scores
        """
        gray = self.preprocess_image(image)
        pts, desc, scores = self.superpoint.run(gray)
        return pts, desc, scores

    def match_descriptors(self, desc1, desc2):
        """
        Match descriptors using nearest neighbor with ratio test.

        Args:
            desc1, desc2: Descriptor matrices (256xN)

        Returns:
            matches: List of (idx1, idx2) matches
        """
        desc1 = desc1.T  # Nx256
        desc2 = desc2.T  # Mx256

        desc1 = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
        desc2 = desc2 / np.linalg.norm(desc2, axis=1, keepdims=True)

        similarity = desc1 @ desc2.T  # NxM

        matches = []
        for i in range(similarity.shape[0]):
            top_matches = np.argsort(similarity[i])[-2:][::-1]
            best_idx = top_matches[0]
            second_best_idx = top_matches[1]

            best_score = similarity[i, best_idx]
            second_best_score = similarity[i, second_best_idx]

            if best_score > self.nn_thresh * second_best_score:
                matches.append((i, best_idx))

        return matches

    def match_boards(self, test_img_path, ideal_img_path, visualize=True):
        """
        Complete pipeline: detect, match, compute homography.

        Args:
            test_img_path: Path to test board image
            ideal_img_path: Path to ideal template
            visualize: Create visualization

        Returns:
            dict with results
        """
        test_img  = cv2.imread(str(test_img_path))
        ideal_img = cv2.imread(str(ideal_img_path))

        if test_img is None or ideal_img is None:
            raise ValueError("Could not load images!")

        print(f"\nProcessing:")
        print(f"   Test:  {Path(test_img_path).name} ({test_img.shape})")
        print(f"   Ideal: {Path(ideal_img_path).name} ({ideal_img.shape})")

        print("\nDetecting keypoints...")
        pts_test,  desc_test,  scores_test  = self.detect_and_compute(test_img)
        pts_ideal, desc_ideal, scores_ideal = self.detect_and_compute(ideal_img)

        print(f"   Test:  {pts_test.shape[1]} keypoints")
        print(f"   Ideal: {pts_ideal.shape[1]} keypoints")

        print("\nMatching descriptors...")
        matches = self.match_descriptors(desc_ideal, desc_test)
        print(f"   Found {len(matches)} matches")

        if len(matches) < 10:
            print("Not enough matches!")
            return None

        pts_ideal_matched = np.array([pts_ideal[:, m[0]] for m in matches])
        pts_test_matched  = np.array([pts_test[:,  m[1]] for m in matches])

        print("\nComputing homography...")
        H, mask = cv2.findHomography(pts_ideal_matched, pts_test_matched, cv2.RANSAC, 5.0)

        if H is None:
            print("Homography computation failed!")
            return None

        inliers = np.sum(mask)
        print(f"   RANSAC inliers: {inliers}/{len(matches)}")

        h, w = test_img.shape[:2]
        warped_ideal = cv2.warpPerspective(ideal_img, H, (w, h))
        overlay = cv2.addWeighted(test_img, 0.5, warped_ideal, 0.5, 0)

        results = {
            'homography': H,
            'warped_ideal': warped_ideal,
            'overlay': overlay,
            'test_image': test_img,
            'ideal_image': ideal_img,
            'pts_test': pts_test,
            'pts_ideal': pts_ideal,
            'scores_test': scores_test,
            'scores_ideal': scores_ideal,
            'matches': matches,
            'mask': mask,
            'num_matches': len(matches),
            'num_inliers': int(inliers)
        }

        if visualize:
            self._visualize_results(results, Path(test_img_path))

        return results

    def _visualize_results(self, results, test_img_path):
        """Create visualization of SuperPoint matching results."""
        fig = plt.figure(figsize=(20, 15))

        ax1 = plt.subplot(3, 3, 1)
        img_with_kp_ideal = self._draw_keypoints(
            results['ideal_image'], results['pts_ideal'], results['scores_ideal']
        )
        ax1.imshow(cv2.cvtColor(img_with_kp_ideal, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Ideal Template\n{results["pts_ideal"].shape[1]} keypoints',
                      fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(3, 3, 2)
        img_with_kp_test = self._draw_keypoints(
            results['test_image'], results['pts_test'], results['scores_test']
        )
        ax2.imshow(cv2.cvtColor(img_with_kp_test, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Test Image\n{results["pts_test"].shape[1]} keypoints',
                      fontsize=12, fontweight='bold')
        ax2.axis('off')

        ax3 = plt.subplot(3, 3, 3)
        match_img = self._draw_matches(results)
        ax3.imshow(match_img)
        ax3.set_title(f'Matches: {results["num_inliers"]}/{results["num_matches"]} inliers',
                      fontsize=12, fontweight='bold')
        ax3.axis('off')

        ax4 = plt.subplot(3, 3, 4)
        ax4.imshow(cv2.cvtColor(results['test_image'], cv2.COLOR_BGR2RGB))
        ax4.set_title('Test Image (Original)', fontsize=12, fontweight='bold')
        ax4.axis('off')

        ax5 = plt.subplot(3, 3, 5)
        ax5.imshow(cv2.cvtColor(results['warped_ideal'], cv2.COLOR_BGR2RGB))
        ax5.set_title('Warped Ideal Template', fontsize=12, fontweight='bold')
        ax5.axis('off')

        ax6 = plt.subplot(3, 3, 6)
        ax6.imshow(cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB))
        ax6.set_title('Alignment Overlay', fontsize=12, fontweight='bold')
        ax6.axis('off')

        ax7 = plt.subplot(3, 3, 7)
        diff = cv2.absdiff(results['test_image'], results['warped_ideal'])
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        im = ax7.imshow(diff_gray, cmap='hot')
        ax7.set_title('Difference Map', fontsize=12, fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im, ax=ax7, fraction=0.046)

        ax8 = plt.subplot(3, 3, 8)
        ax8.hist(results['scores_test'],  bins=50, alpha=0.7, label='Test')
        ax8.hist(results['scores_ideal'], bins=50, alpha=0.7, label='Ideal')
        ax8.set_xlabel('Confidence Score')
        ax8.set_ylabel('Count')
        ax8.set_title('Keypoint Confidence Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        stats_text = f"""
SuperPoint Matching Results

Detection:
  Ideal keypoints: {results['pts_ideal'].shape[1]}
  Test keypoints: {results['pts_test'].shape[1]}
  Avg confidence (ideal): {results['scores_ideal'].mean():.3f}
  Avg confidence (test): {results['scores_test'].mean():.3f}

Matching:
  Total matches: {results['num_matches']}
  RANSAC inliers: {results['num_inliers']}
  Inlier ratio: {results['num_inliers']/results['num_matches']:.1%}

Homography:
{np.array2string(results['homography'], precision=3, suppress_small=True)}
        """
        ax9.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
                 verticalalignment='center')

        plt.tight_layout()

        results_dir = os.getenv('RESULTS_DIR', str(test_img_path.parent))
        output_path = Path(results_dir) / f"{test_img_path.stem}_superpoint_results.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.show()

    def _draw_keypoints(self, image, pts, scores, top_k=500):
        """Draw keypoints on image with confidence-based coloring."""
        img = image.copy()
        n_keypoints = min(pts.shape[1], len(scores))
        pts_draw    = pts[:, :n_keypoints]
        scores_draw = scores[:n_keypoints]

        if n_keypoints > top_k:
            top_indices = np.argsort(scores_draw)[-top_k:]
            pts_draw    = pts_draw[:, top_indices]
            scores_draw = scores_draw[top_indices]
            n_keypoints = top_k

        for i in range(n_keypoints):
            x, y       = int(pts_draw[0, i]), int(pts_draw[1, i])
            confidence = scores_draw[i]
            color      = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.circle(img, (x, y), 3, color, -1)

        return img

    def _draw_matches(self, results):
        """Draw lines between matched keypoints."""
        h1, w1 = results['ideal_image'].shape[:2]
        h2, w2 = results['test_image'].shape[:2]

        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1]       = results['ideal_image']
        canvas[:h2, w1:w1+w2]  = results['test_image']

        for i, (idx1, idx2) in enumerate(results['matches']):
            if results['mask'][i]:
                pt1   = (int(results['pts_ideal'][0, idx1]),
                         int(results['pts_ideal'][1, idx1]))
                pt2   = (int(results['pts_test'][0, idx2] + w1),
                         int(results['pts_test'][1, idx2]))
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.line(canvas, pt1, pt2, color, 1)
                cv2.circle(canvas, pt1, 3, color, -1)
                cv2.circle(canvas, pt2, 3, color, -1)

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def main():
    """Run SuperPoint matching on board game dataset."""
    weights_path  = os.getenv('SP_WEIGHTS', _SP_WEIGHTS)
    template_dir  = os.getenv('TEMPLATE_DIR', '')
    test_image_dir = os.getenv('TEST_IMAGE_DIR', '')

    if not Path(weights_path).exists():
        print(f"Model weights not found at: {weights_path}")
        print("Set SP_WEIGHTS in superpoint/.env")
        return

    matcher = SuperPointBoardMatcher(
        weights_path=weights_path,
        conf_thresh=0.015,
        nn_thresh=0.7
    )

    test_images  = sorted(Path(test_image_dir).glob('*.png'))  + sorted(Path(test_image_dir).glob('*.jpg'))
    ideal_images = sorted(Path(template_dir).glob('*.png')) + sorted(Path(template_dir).glob('*.jpg'))

    if not test_images or not ideal_images:
        print("No images found. Check TEMPLATE_DIR and TEST_IMAGE_DIR in .env")
        return

    results = matcher.match_boards(test_images[0], ideal_images[0])

    if results:
        print(f"\nMatching completed | matches: {results['num_matches']} | inliers: {results['num_inliers']}")
    else:
        print("Matching failed!")


if __name__ == "__main__":
    main()
