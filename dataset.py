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
                A.CoarseDropout(max_holes=50, max_height=20, max_width=20),
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

        # random crop
        off_x, off_y = np.random.randint(0, 30, 2)
        img = img[off_y:, off_x:]
        h, w = img.shape[:2]
        img = img[:h // 32 * 32, :w // 32 * 32]
        h, w = img.shape[:2]
        x, y = x - off_x, y - off_y

        # TODO: random overlay
        # TODO: continuous rotation augmentation
        r = np.random.randint(0, 4)
        theta += r * .5 * np.pi
        if r == 1:
            img = img.transpose(1, 0, 2)[:, ::-1]
            x, y = h - y - 1, x
        elif r == 2:
            img = img[::-1, ::-1]
            x, y = w - x - 1, h - y - 1
        elif r == 3:
            img = img.transpose(1, 0, 2)[::-1]
            x, y = y, w - x - 1

        img = self.tfms(image=img)['image']
        while theta < 0:
            theta += np.pi * 2
        while theta > np.pi * 2:
            theta -= np.pi * 2
        return img, x, y, theta


def _main():
    import vis
    name = 'big_pulley'
    rgba_template = utils.load_rgba_template(name)
    dataset = Dataset(name, norm=True)
    img = dataset[0][0]  # (C, H, W)
    print(img.mean(dim=(1, 2)), img.std(dim=(1, 2)))

    dataset = Dataset(name, norm=False)
    while True:
        img, *pose = dataset[np.random.randint(len(dataset))]
        img_overlay = vis.overlay_template(img, rgba_template, *pose)
        cv2.imshow('', img_overlay)
        key = cv2.waitKey()
        if key == ord('q'):
            return


if __name__ == '__main__':
    _main()
