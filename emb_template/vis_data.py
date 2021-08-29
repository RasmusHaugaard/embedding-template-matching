import argparse

import cv2

from . import vis
from . import utils
from .dataset import Dataset
from .camera import CameraInfo

parser = argparse.ArgumentParser()
parser.add_argument('object_name')
args = parser.parse_args()
object_name = args.object_name

rgba_template, obj_t_template, sym = utils.load_current_template(object_name)
annotation_fps, image_fps = utils.load_valid_annotation_and_image_fps(object_name)
cam_info = CameraInfo.load()
dataset = Dataset(annotation_fps=annotation_fps, image_fps=image_fps,
                  obj_t_template=obj_t_template, K=cam_info.K, norm=False)

print('"a" and "d" to go through images.\n'
      'Space or Enter to get new instance of same image.\n'
      '"h" to hide / unhide template\n'
      'Delete or Backspace to ')

i, hide = 0, False
img, pose = None, None


def draw(update_image=True):
    global img, pose, hide
    if update_image:
        img, *pose = dataset[i]
        hide = False
    if not hide:
        cv2.imshow('', vis.overlay_template(img, rgba_template, *pose))
    else:
        cv2.imshow('', img)


draw()
while True:
    key = cv2.waitKey()
    if key == ord('q'):
        break
    if key == ord('a'):
        i = (i - 1) % len(dataset)
        draw()
    elif key == ord('d'):
        i = (i + 1) % len(dataset)
        draw()
    elif key == ord('h'):
        hide = not hide
        draw(update_image=False)
    elif key == ord('\r') or key == ord(' '):
        draw()
