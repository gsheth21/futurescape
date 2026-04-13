from preprocessing import images_to_tensors
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../SuperPointPretrainedNetwork'))
from demo_superpoint import SuperPointFrontend


WEIGHTS     = '../SuperPointPretrainedNetwork/pretrained/superpoint_v1.pth'
NMS_DIST    = 4
CONF_THRESH = 0.015
NN_THRESH   = 0.7


def load_model(weights_path=WEIGHTS, cuda=False):
    """Load pretrained SuperPoint model."""
    cuda = cuda and torch.cuda.is_available()
    fe = SuperPointFrontend(
        weights_path=weights_path,
        nms_dist=NMS_DIST,
        conf_thresh=CONF_THRESH,
        nn_thresh=NN_THRESH,
        cuda=cuda
    )
    print(f"SuperPoint loaded | GPU={'yes' if cuda else 'no'}")
    return fe


def detect(fe, tensor):
    """
    Run SuperPoint on a (1,1,H,W) float32 tensor.
    Returns:
        pts   : (3, N)   x, y, confidence
        desc  : (256, N)
        scores: (N,)
    """
    img_np = tensor.squeeze().numpy()
    pts, desc, scores = fe.run(img_np)
    return pts, desc, scores