import cv2
import numpy as np
import torch
from transform3d import Transform

import utils


def show_template(model, rotate=False):
    emb_template = model.get_template().detach().cpu().numpy()
    mi, ma = emb_template.min(), emb_template.max()
    emb_template = (emb_template - mi) / (ma - mi)
    for i in range(len(emb_template) if rotate else 1):
        im = emb_template[i, :3].transpose((1, 2, 0)).copy()
        # print(im.shape)
        cv2.imshow('', im)
        cv2.waitKey()


def overlay_activation_2d(img, act, stride):
    hi, wi = img.shape[:2]
    act = act.max(dim=0)[0]  # (h, w)
    h, w = act.shape
    act = torch.softmax(act.view(-1), 0).view(*act.shape)
    act = (act.cpu().numpy() * 255).round().astype(np.uint8)
    act = np.stack((np.zeros_like(act), np.zeros_like(act), act), axis=-1)
    hh, ww = h * stride, w * stride
    act = cv2.resize(act, (ww, hh), interpolation=cv2.INTER_LINEAR)
    h, w = min(hi, hh), min(wi, ww)
    temp = img[:h, :w] // 2 + act[:h, :w] // 2
    return temp


def overlay_template(img, rgba_template, x, y, theta):
    h, w = img.shape[:2]
    k = rgba_template.shape[0] // 2
    temp = img.copy()
    M = Transform(p=(x, y, 0)) @ Transform(rotvec=(0, 0, theta)) @ Transform(p=(-k, -k, 0))
    M = M.matrix[:2, (0, 1, 3)]
    template_vis = rgba_template.copy()
    template_vis[..., 2] = np.clip((template_vis[..., 2].astype(int) * 3) // 2, 0, 255)
    over = cv2.warpAffine(template_vis, M, (w, h))
    mask = over[..., 3] == 255
    temp[mask] = temp[mask] // 2 + over[..., :3][mask] // 2
    return temp


@torch.no_grad()
def _main():
    import argparse
    from model import Model
    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument('--template', action='store_true')
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    model = Model.load_from_checkpoint(
        checkpoint_path=utils.latest_checkpoint(),
        rgba_template=cv2.imread('templates/big_pulley.png', cv2.IMREAD_UNCHANGED), normalize=False,
    )

    if args.template:
        show_template(model)

    if args.infer:
        model.to(device)
        for i in range(5, 7):
            img = cv2.imread(f'images/{i}.png')[:704]
            img_ = utils.normalize(img)[None].to(model.device)
            act = model.forward(img_)[0]  # (angle_thresh, h, w)
            act_img = overlay_activation_2d(img, act, model.stride)
            cv2.imshow('act', act_img)

            if cv2.waitKey() == ord('q'):
                break


if __name__ == '__main__':
    _main()
