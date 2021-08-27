import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation

from . import unet


class Model(pl.LightningModule):
    def __init__(self, rgba_template: np.ndarray, emb_dim=3, normalize=False, angle_res=45):
        super().__init__()
        self.s = rgba_template.shape[0]
        self.angle_res = angle_res
        self.k = self.s // 2
        self.normalize = normalize
        self.emb_dim = emb_dim
        self.model = unet.ResNetUNet(emb_dim)
        if normalize and False:
            self.template = torch.nn.Parameter(torch.randn(1, emb_dim, self.s, self.s))
        else:
            self.template = torch.nn.Parameter(torch.zeros(1, emb_dim, self.s, self.s))
        self.register_buffer('mask', torch.from_numpy(rgba_template[..., 3] == 255).float())
        self.stride = max(1, self.s // 20)

        rotvecs = np.linspace(0, -2 * np.pi, angle_res, endpoint=False)
        rotvecs = np.stack((np.zeros(angle_res), np.zeros(angle_res), rotvecs), axis=-1)  # (N, 3)
        M = np.zeros((angle_res, 3, 3))
        M[:, 2, 2] = 1
        M[:, :2, :2] = Rotation.from_rotvec(rotvecs).as_matrix()[:, :2, :2]
        M = torch.from_numpy(M[:, :2].astype(np.float32))
        self.register_buffer('M', M)
        self.pad = 'reflect'

    def emb_forward(self, img):
        emb = self.model(img)  # (B, C, H, W)
        if self.normalize:
            emb = F.normalize(emb, dim=1)
        return emb

    def get_template(self):
        t = self.template  # (1, emb_dim, s, s)
        t = t * self.mask[None, None]
        if self.normalize and False:
            t = F.normalize(t, dim=1)
        t = torch.broadcast_to(t, (self.angle_res, self.emb_dim, self.s, self.s))
        grid = F.affine_grid(self.M, [self.angle_res, self.emb_dim, self.s, self.s], align_corners=False)
        t = F.grid_sample(t, grid, align_corners=False)  # (angle_res, emb_dim, s, s)
        return t

    def forward(self, img):
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
        theta_idx = torch.round(theta / (np.pi * 2 / self.angle_res)).long()
        act_flat = act.view(b, self.angle_res * h * w)
        target = theta_idx * h * w + y * w + x
        ignore_mask = torch.logical_or(target < 0, self.angle_res * h * w <= target)
        target[ignore_mask] = -1
        loss = F.cross_entropy(act_flat, target, ignore_index=-1)
        self.log(f'{name}_loss', loss, prog_bar=prog_bar)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        return self.step(batch, 'val', prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
