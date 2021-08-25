from pathlib import Path

import natsort
import albumentations as A
from albumentations.pytorch import ToTensorV2

_normalize = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


def normalize(img):
    return _normalize(image=img)['image']


def sorted_paths(paths):
    return natsort.natsorted(paths, key=lambda path: str(path))


def latest_checkpoint():
    fp = sorted_paths(Path('lightning_logs').glob('version_*'))[-1]
    fp = sorted_paths((fp / 'checkpoints').glob('*.ckpt'))[-1]
    return fp
