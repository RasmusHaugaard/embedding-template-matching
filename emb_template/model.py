import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation

from . import unet
from . import utils


class Model(pl.LightningModule):
    def __init__(self, rgba_template: np.ndarray, sym: int,
                 emb_dim=3, beta: float = 0.,
                 angle_resolution=90, position_resolution=30,
                 img_scale=1.):
        super().__init__()
        self.save_hyperparameters('emb_dim', 'angle_resolution', 'position_resolution', 'img_scale')
        self.emb_dim = emb_dim
        self.img_scale = img_scale
        rgba_template = utils.resize(rgba_template, img_scale, interp=cv2.INTER_NEAREST)[0]
        self.s = rgba_template.shape[0]
        self.sym = sym
        self.beta = beta
        if sym == -1:
            self.local_angle_res = 1
            self.local_angle_span = 1
        else:
            self.local_angle_res = int(np.ceil(angle_resolution / sym))
            self.local_angle_span = 2 * np.pi / sym
        self.k = self.s // 2
        self.model = unet.ResNetUNet(emb_dim)
        self.template = torch.nn.Parameter(torch.zeros(1, emb_dim, self.s, self.s))
        self.register_buffer('mask', torch.from_numpy(rgba_template[..., 3] == 255).float())
        self.stride = max(1, self.s // position_resolution)

        rotvecs = np.stack((
            np.zeros(self.local_angle_res),
            np.zeros(self.local_angle_res),
            np.linspace(0, -2 * np.pi / sym, self.local_angle_res, endpoint=False)
        ), axis=-1)  # (N, 3)
        M = np.zeros((self.local_angle_res, 3, 3))
        M[:, 2, 2] = 1
        M[:, :2, :2] = Rotation.from_rotvec(rotvecs).as_matrix()[:, :2, :2]
        M = torch.from_numpy(M[:, :2].astype(np.float32))
        self.register_buffer('M', M)
        self.pad = 'reflect'

    def emb_forward(self, img):
        emb = self.model(img)  # (B, C, H, W)
        return emb

    def get_template(self):
        t = self.template  # (1, emb_dim, s, s)
        t = t * self.mask[None, None]
        t = torch.broadcast_to(t, (self.local_angle_res, self.emb_dim, self.s, self.s))
        grid = F.affine_grid(self.M, [self.local_angle_res, self.emb_dim, self.s, self.s], align_corners=False)
        t = F.grid_sample(t, grid, align_corners=False)  # (angle_res, emb_dim, s, s)
        return t

    def forward(self, img):
        h, w = img.shape[-2:]
        # unet can only handle resolutions in multiples of 32
        h, w = h // 32 * 32, w // 32 * 32
        img = img[..., :h, :w]
        emb = self.emb_forward(img)  # (B, C, H, W)
        if self.pad is not None:
            emb = F.pad(emb, [self.k] * 4, mode=self.pad)
        # conv2d is actually cross-correlation
        act = F.conv2d(emb, self.get_template(), stride=self.stride)  # (B, angle_res, h, w)
        return act, emb

    def step(self, batch, name, prog_bar=False):
        img, x, y, theta = batch
        act = self.forward(img)[0]
        b, _, h, w = act.shape
        offset = self.k if self.pad is None else 0
        x = torch.round((x - offset) / self.stride).long()
        y = torch.round((y - offset) / self.stride).long()
        theta_idx = torch.round(theta / self.local_angle_span * self.local_angle_res).long() % self.local_angle_res
        act_flat = act.view(b, self.local_angle_res * h * w)
        target = theta_idx * h * w + y * w + x
        loss = F.cross_entropy(act_flat, target)

        if self.beta > 0:
            laplace_kernel = torch.broadcast_to(torch.tensor([
                [1, 1, 1], [1, -8., 1], [1, 1, 1]
            ]).view(1, 1, 3, 3), (self.emb_dim, 1, 3, 3)).to(self.device)
            laplace_act = F.conv2d(self.template, laplace_kernel, groups=self.emb_dim)
            loss += self.beta * (laplace_act ** 2).mean()

        self.log(f'{name}_loss', loss, prog_bar=prog_bar)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        return self.step(batch, 'val', prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        def lr(i):
            # warmup
            lr_ = min(i / 40, 1.)
            return lr_

        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lr),
                interval='step',
            ),
        )
