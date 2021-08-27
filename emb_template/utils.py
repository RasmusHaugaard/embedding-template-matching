import json
from pathlib import Path

import cv2
import numpy as np
import natsort
import albumentations as A
from albumentations.pytorch import ToTensorV2

_normalize = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


def premultiply_alpha(img):
    mask, img = img[..., 3:], img[..., :3]
    img = (img * mask.astype(np.uint16)) // 255
    return img.astype(np.uint8)


def normalize(img):
    return _normalize(image=img)['image']


def sorted_paths(paths):
    return natsort.natsorted(paths, key=lambda path: str(path))


def latest_checkpoint():
    fp = sorted_paths(Path('lightning_logs').glob('version_*'))[-1]
    fp = sorted_paths((fp / 'checkpoints').glob('*.ckpt'))[-1]
    return fp


def pose_from_act(act: np.ndarray, stride: int):
    theta, y, x = np.unravel_index(np.argmax(act), act.shape)
    y, x = y * stride, x * stride
    theta = theta / len(act) * 2 * np.pi
    return x, y, theta


def load_rgba_template(name):
    t = cv2.imread(f'rgba_templates/{name}.png', cv2.IMREAD_UNCHANGED)
    if t is None:
        raise FileExistsError('could not find template')
    return t
