from pathlib import Path

import cv2
import numpy as np
import torch.utils.data
import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, name='big_pulley', data_slice=slice(None), tfms=True, norm=True):
        self.annotation_fps = utils.sorted_paths(Path('annotations').glob(f'*.{name}.txt'))[data_slice]
        self.img_fps = [f"images/{str(fp.name).split('.')[0]}.png" for fp in self.annotation_fps]
        self.tfms = []
        if tfms:
            self.tfms += [
                A.ColorJitter(hue=0.1),
                A.ISONoise(),
                A.GaussNoise(),
                A.GaussianBlur(),
            ]
        if norm:
            self.tfms += [
                A.Normalize(),
                ToTensorV2(),
            ]
        self.tfms = A.Compose(self.tfms)

    def __len__(self):
        return len(self.annotation_fps)

    def __getitem__(self, i):
        x, y, theta = np.loadtxt(str(self.annotation_fps[i]))
        img = cv2.imread(str(self.img_fps[i]))

        off_x, off_y = np.random.randint(0, 30, 2)
        img = img[off_y:, off_x:]
        h, w = img.shape[:2]
        img = img[:h // 32 * 32, :w // 32 * 32]
        x, y = x - off_x, y - off_y

        # TODO: better data augs (translation, rotation)
        img = self.tfms(image=img)['image']
        while theta < 0:
            theta += np.pi * 2
        while theta > np.pi * 2:
            theta -= np.pi * 2
        return img, x, y, theta


def _main():
    dataset = Dataset('big_pulley', norm=True)
    img = dataset[0][0]  # (C, H, W)
    print(img.mean(dim=(1, 2)), img.std(dim=(1, 2)))

    dataset = Dataset('big_pulley', norm=False)
    while True:
        img = dataset[0][0]
        cv2.imshow('', img)
        key = cv2.waitKey()
        if key == ord('q'):
            return


if __name__ == '__main__':
    _main()
