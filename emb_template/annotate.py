import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from transform3d import Transform
import rospy
import tqdm

from . import vis
from . import utils
from . import camera
from .renderer import MeshRenderer

parser = argparse.ArgumentParser()
parser.add_argument('object_name')
parser.add_argument('--oldest-first', action='store_true')
parser.add_argument('--translation-scaling', type=float, default=0.3)
parser.add_argument('--rotation-scaling', type=float, default=0.008)
args = parser.parse_args()

obj_folder = Path('objects') / args.object_name
mesh = trimesh.load_mesh(obj_folder / 'cad.stl')
cam_info = camera.CameraInfo.load()
K = cam_info.K
annotation_folder = obj_folder / 'annotations'
annotation_folder.mkdir(exist_ok=True)

annotations_required = set(utils.load_image_names()) - set(utils.load_annotation_names(args.object_name))
annotations_required = sorted(annotations_required, reverse=not args.oldest_first)
if not annotations_required:
    print(f'No images need "{args.object_name}" annotations')
    quit()

renderer = MeshRenderer(mesh=mesh, h=cam_info.h, w=cam_info.w, K=cam_info.K)
rospy.init_node('annotate', anonymous=True)
cam = camera.Camera(cam_info.image_topic)
cam_t_table = Transform.load('cam_t_table.txt')
table_normal = cam_t_table.R[:, 2]
table_point = cam_t_table.p
table_t_obj_stable = Transform.load(obj_folder / 'table_t_obj_stable.txt')

cam_t_table_center = utils.get_cam_t_table_center(cam_t_table=cam_t_table, K=K, w=cam_info.w, h=cam_info.h)
# TODO: center based on CAD model bounding xy circle (cv2.minEnclosingCircle)
cam_t_obj_init = cam_t_table_center @ table_t_obj_stable
# TODO: define the pose in table xy plane during pose annotation to avoid drifting

print()
print('Annotate pose by moving the mouse while holding ctrl (pos) and shift (rotation).\n'
      'Press \n'
      '  - Enter or Space to confirm pose \n'
      '  - Delete or Backspace if the object is not in the image\n'
      '  - r to reset pose\n'
      '  - q to quit')

for image_name in tqdm.tqdm(annotations_required):
    annotation_path = obj_folder / 'annotations' / f'{image_name}.txt'
    img = cv2.imread(f'images/{image_name}.png')
    cam_t_obj = cam_t_obj_init
    mx, my = None, None  # mouse position
    hidden = False


    def draw():
        global hidden
        render, _ = renderer.render(cam_t_obj)
        cv2.imshow('', vis.composite(img, render[..., :3], render[..., 3:] // 2))
        hidden = False


    def mouse_cb(event, x, y, flags, _):
        global mx, my, cam_t_obj
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if mx is not None and my is not None:
                dp_img = np.array((x - mx, y - my))
                jac_img_xy = K[:2, :2] / cam_t_obj.p[2]
                jac_img_table = jac_img_xy @ cam_t_table.R[:2, :2]
                # jac_img_table @ dp_table = dp_img, solve for dp_table:
                dp_table = np.linalg.solve(jac_img_table, dp_img)
                dp_cam = cam_t_table.R @ (*dp_table, 0)
                cam_t_obj = Transform(p=args.translation_scaling * dp_cam) @ cam_t_obj
                draw()
            mx, my = x, y
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            if mx is not None and my is None:
                # TODO: rotate as if the mouse is dragging the object around the rotation axis
                phi = (x - mx) * args.rotation_scaling
                cam_t_obj = cam_t_obj @ table_t_obj_stable.inv @ Transform(rotvec=(0, 0, phi)) @ table_t_obj_stable
                draw()
            mx, my = x, None
        else:
            mx, my = None, None


    draw()
    cv2.setMouseCallback('', mouse_cb)
    while True:
        key = cv2.waitKey()
        if key == ord('q'):
            quit()
        elif key == ord(' ') or key == ord('\r'):
            cam_t_obj.save(annotation_path)
            break
        elif key == ord('s'):
            break
        elif key == 8 or key == 255:  # Backspace, Delete
            annotation_path.open('w').close()
            break
        elif key == ord('r'):
            cam_t_obj = cam_t_obj_init
        elif key == ord('h'):
            hidden = not hidden
            if hidden:
                cv2.imshow('', img)
            else:
                draw()
