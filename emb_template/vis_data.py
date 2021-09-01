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

rgba_template, _, obj_t_template, sym = utils.load_current_template(object_name)
annotation_fps, image_fps = utils.load_valid_annotation_and_image_fps(object_name)
cam_info = CameraInfo.load()


def get_dataset():
    return Dataset(annotation_fps=annotation_fps, image_fps=image_fps,
                   obj_t_template=obj_t_template, K=cam_info.K, norm=False)


print('"a" and "d" to go through images.\n'
      'Space or Enter to get new instance of same image.\n'
      '"h" to hide / unhide template\n'
      'Delete or Backspace to delete annotation')

i, hide = 0, False
img, pose = None, None
dataset = get_dataset()


def draw(update_image=True):
    global img, pose, hide
    if update_image:
        img, *pose = dataset[i]
        hide = False
    img_temp = img.copy()
    cv2.putText(img_temp, str(image_fps[i]), (0, 12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    if not hide:
        cv2.imshow('', vis.overlay_template(img_temp, rgba_template, *pose))
    else:
        cv2.imshow('', img_temp)


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
    elif key == 8 or key == 255:  # Backspace, Delete
        annotation_fps[i].unlink()
        annotation_fps.pop(i)
        image_fps.pop(i)
        if not image_fps:
            quit()
        dataset = get_dataset()
        i = i % len(dataset)
        draw()
    print(f'{i + 1} / {len(dataset)}')
