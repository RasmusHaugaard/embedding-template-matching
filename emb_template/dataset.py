import cv2
import numpy as np
import torch.utils.data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transform3d import Transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_fps, annotation_fps, obj_t_template: Transform,
                 K: np.ndarray, tfms=True, norm=True):
        assert len(image_fps) == len(annotation_fps)
        self.image_fps, self.annotation_fps = image_fps, annotation_fps
        self.obj_t_template = obj_t_template
        self.K = K

        self.tfms = []
        if tfms:
            self.tfms += [
                A.CoarseDropout(max_holes=50, max_height=20, max_width=20),
                A.ColorJitter(brightness=0.4, contrast=0.4, hue=0.1),
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
        img = cv2.imread(str(self.image_fps[i]))
        cam_t_obj = Transform.load(self.annotation_fps[i])
        cam_t_template = cam_t_obj @ self.obj_t_template
        line = self.K @ cam_t_template.p
        x, y = line[:2] / line[2]
        # theta is the rotation around the camera principal axis.
        img_xaxis_template = cam_t_template.R[:2, 0]
        theta = np.arctan2(*img_xaxis_template[::-1])

        # random crop
        off_x, off_y = np.random.randint(0, 30, 2)
        img = img[off_y:, off_x:]
        h, w = img.shape[:2]
        img = img[:h // 32 * 32, :w // 32 * 32]
        h, w = img.shape[:2]
        x, y = x - off_x, y - off_y

        # continuous rotation augmentation
        theta_off = np.random.uniform(-45, 45)
        M = cv2.getRotationMatrix2D((x, y), theta_off, 1)
        img = cv2.warpAffine(src=img, M=M, dsize=(w, h), borderMode=cv2.BORDER_REFLECT)
        theta -= theta_off * np.pi / 180

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
