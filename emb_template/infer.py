import time
import argparse
import datetime

import cv2
import numpy as np
import torch
import rospy
from transform3d import Transform

from . import utils
from . import vis
from .camera import Camera, CameraInfo
from .model import Model
from .renderer import MeshRenderer

parser = argparse.ArgumentParser()
parser.add_argument('object_name')
parser.add_argument('--show-template', action='store_true')
parser.add_argument('--show-embedding', action='store_true')
parser.add_argument('--show-activation', action='store_true')
parser.add_argument('--show-certainty', action='store_true')
parser.add_argument('--show-angle-dist', action='store_true')
args = parser.parse_args()
object_name = args.object_name

cam_t_table = Transform.load('cam_t_table.txt')
rgba_template, table_offset, obj_t_template, sym = utils.load_current_template(object_name)
cam_info = CameraInfo.load()
renderer = MeshRenderer(
    mesh=utils.load_mesh(object_name), h=cam_info.h, w=cam_info.w, K=cam_info.K,
)

rospy.init_node('infer', anonymous=True)

model = Model.load_from_checkpoint(
    # TODO: load *current* model
    utils.latest_checkpoint(object_name),
    rgba_template=rgba_template, sym=sym
)
model.eval()
model.cuda()

cam = Camera()

if args.show_template:
    template = np.concatenate((
        vis.premultiply_alpha(rgba_template),
        vis.emb_for_vis(model.get_template()[0]),
    ), axis=1)
    cv2.imshow('template', template)

print('Press Space or Enter to save the current image for annotation.\n'
      '"r" to switch between 3D pose render and template overlay')

do_render = True

while True:
    img = cam.take_image()
    img_ = utils.normalize(img).to(model.device)
    with torch.no_grad():
        act, emb = model.forward(img_[None])
        act, emb = act[0], emb[0]
    if args.show_activation:
        cv2.imshow('act', vis.overlay_activation_2d(img, act, model.stride))
    if args.show_embedding:
        cv2.imshow('emb', vis.emb_for_vis(emb))
    probs = torch.softmax(act.view(-1), 0).view(*act.shape)
    certainty = probs.max().item()

    act = act.cpu().numpy()
    pose_2d = utils.pose_2d_from_act(act=act, stride=model.stride, sym=sym)
    cam_t_obj = utils.get_pose_3d(
        pose_2d=pose_2d, K=cam_info.K, cam_t_table=cam_t_table,
        table_offset=table_offset, obj_t_template=obj_t_template,
    )

    if do_render:
        render = renderer.render(cam_t_obj)[0].copy()
        render[..., :2] = 0
        img_overlay = vis.composite(img, render[..., :3], render[..., 3:] // 2)
    else:
        img_overlay = vis.overlay_template(img, rgba_template, *pose_2d)
    cv2.putText(img_overlay, f'certainty: {certainty:.2f}', (0, 12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    cv2.imshow('', img_overlay)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('\r') or key == ord(' '):
        cv2.imwrite(str(utils.get_image_folder() / f'{datetime.datetime.now()}.png'), img)
        print('Image saved.')
        time.sleep(.3)
    elif key == ord('r'):
        do_render = not do_render
