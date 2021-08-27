import time

import cv2
import torch
import rospy

from . import utils
from . import vis
from .camera import Camera
from .model import Model

name = 'big_pulley'

rospy.init_node('infer', anonymous=True)

rgba_template = utils.load_rgba_template(name)
model = Model.load_from_checkpoint(
    utils.latest_checkpoint(), rgba_template=rgba_template
)
model.eval()
model.cuda()

cam = Camera()
cv2.imshow('rgba_template', vis.premultiply_alpha(rgba_template))
cv2.imshow('emb_template', vis.emb_for_vis(model.get_template()[0]))

while True:
    img = cam.take_image()
    img_ = utils.normalize(img)[None].to(model.device)
    with torch.no_grad():
        act, emb = model.forward(img_)
    cv2.imshow('act', vis.overlay_activation_2d(img, act[0], model.stride))
    act = act[0].cpu().numpy()

    pose = utils.pose_from_act(act, model.stride)
    img_overlay = vis.overlay_template(img, rgba_template, *pose)
    cv2.imshow('', img_overlay)
    cv2.imshow('emb', vis.emb_for_vis(emb[0]))
    if cv2.waitKey(1) == ord('q'):
        break
