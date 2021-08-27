import cv2
import numpy as np
import torch
from transform3d import Transform


def premultiply_alpha(img):
    mask, img = img[..., 3:], img[..., :3]
    img = (img * mask.astype(np.uint16)) // 255
    return img.astype(np.uint8)


def composite(img, overlay, alpha):
    img = img.astype(np.uint16) * (255 - alpha)
    overlay = overlay.astype(np.uint16) * alpha
    return ((img + overlay) // 255).astype(np.uint8)


def emb_for_vis(emb):
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy().transpose((1, 2, 0))
    assert isinstance(emb, np.ndarray)
    emb = emb[..., :3]
    ma = np.abs(emb).max()
    emb = 0.5 + emb / (ma * 2)
    return emb


def show_template(model, rotate=False):
    emb_template = model.get_template()
    for i in range(len(emb_template) if rotate else 1):
        cv2.imshow('', emb_for_vis(emb_template[i]))
        cv2.waitKey()


def overlay_activation_2d(img, act, stride):
    hi, wi = img.shape[:2]
    assert len(act.shape) == 3
    _, h, w = act.shape
    probs = torch.softmax(act.view(-1), 0).view(*act.shape)
    probs_2d = probs.sum(dim=0)  # (h, w)
    probs_2d = torch.round(probs_2d * 255).cpu().numpy().astype(np.uint8)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[..., 2] = probs_2d
    hh, ww = h * stride, w * stride
    overlay = cv2.resize(overlay, (ww, hh), interpolation=cv2.INTER_LINEAR)
    h, w = min(hi, hh), min(wi, ww)
    temp = img[:h, :w] // 2 + overlay[:h, :w] // 2
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
    parser.add_argument('--name', required=True)
    parser.add_argument('--template', action='store_true')
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()

    rgba_template = utils.load_rgba_template(args.name)

    device = torch.device('cuda:0')
    model = Model.load_from_checkpoint(
        checkpoint_path=utils.latest_checkpoint(),
        rgba_template=rgba_template,
    )

    if args.template:
        show_template(model, rotate=True)

    if args.infer:
        model.to(device)
        for i in range(0, 7):
            img = cv2.imread(f'images/{i}.png')[:704]
            img_ = utils.normalize(img)[None].to(model.device)
            act = model.forward(img_)[0][0]  # (angle_thresh, h, w)
            act_img = overlay_activation_2d(img, act, model.stride)
            cv2.imshow('act', act_img)
            pose = utils.pose_from_act(act.cpu().numpy(), model.stride)
            cv2.imshow('pose', overlay_template(img, rgba_template, *pose))
            if cv2.waitKey() == ord('q'):
                break


if __name__ == '__main__':
    _main()
