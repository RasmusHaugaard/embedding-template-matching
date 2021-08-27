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
from . import camera
from .renderer import MeshRenderer

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--oldest-first', action='store_true')
args = parser.parse_args()

obj_folder = Path('objects') / args.name
mesh = trimesh.load_mesh(obj_folder / 'mesh.stl')
cam_info = camera.CameraInfo.load()
K = cam_info.K
annotation_folder = obj_folder / 'annotations'
annotation_folder.mkdir(exist_ok=True)

image_names = set(fp.name[:-4] for fp in Path('images').glob('*.png'))
annotation_names = set(fp.name[:-4] for fp in (obj_folder / 'annotations').glob('*.txt'))
image_names_need_annotation = sorted(list(image_names - annotation_names), reverse=not args.oldest_first)
if not image_names_need_annotation:
    print(f'No images need "{args.name}" annotations')
    quit()

renderer = MeshRenderer(mesh=mesh, h=cam_info.h, w=cam_info.w, K=cam_info.K)
rospy.init_node('annotate', anonymous=True)
cam = camera.Camera(cam_info.image_topic)
cam_t_table = Transform.load('cam_t_table.txt')
table_normal = cam_t_table.R[:, 2]
table_point = cam_t_table.p
table_t_obj_stable = Transform.load(obj_folder / 'table_t_obj_stable.txt')

# find intersection point of center view ray and table plane
# TODO: refactor to utils / math or see if other library does this
ray_direction = np.linalg.inv(K) @ (cam_info.w / 2, cam_info.h / 2, 1)
ray_direction = ray_direction / np.linalg.norm(ray_direction)
ray_point = np.zeros(3)
w = ray_point - table_point
si = -table_normal @ w / (table_normal @ ray_direction)
table_center_p_cam = w + si * ray_direction + table_point

# find pose in middle of image
# TODO: center based on CAD model bounding xy circle
table_center_p_table = cam_t_table.inv @ table_center_p_cam
cam_t_obj_init = cam_t_table @ Transform(p=table_center_p_table) @ table_t_obj_stable
# TODO: define the pose in table xy plane during pose annotation to avoid drifting
print(cam_t_table.R)

print()
print('Annotate pose by moving the mouse while holding ctrl (pos) and shift (rotation).\n'
      'Press \n'
      '  - Space to confirm pose \n'
      '  - Delete if the object is not in the image\n'
      '  - r to reset pose\n'
      '  - q to quit')

for image_name in tqdm.tqdm(image_names_need_annotation):
    annotation_path = obj_folder / 'annotations' / f'{image_name}.txt'
    img = cv2.imread(f'images/{image_name}.png')
    cam_t_obj = cam_t_obj_init
    mx, my = None, None  # mouse position


    def draw():
        render, _ = renderer.render(cam_t_obj)
        cv2.imshow('', vis.composite(img, render[..., :3], render[..., 3:] // 2))


    def get_jac_img_table():
        jac_img_xy = K[:2, :2] / cam_t_obj.p[2]
        jac_img_table = jac_img_xy @ cam_t_table.R[:2, :2]
        return jac_img_table


    def mouse_cb(event, x, y, flags, _):
        global mx, my, cam_t_obj
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if mx is not None and my is not None:
                dp_img = np.array((x - mx, y - my))
                # jac_img_table @ dp_table = dp_img
                dp_table = np.linalg.solve(get_jac_img_table(), dp_img)
                dp_cam = cam_t_table.R @ (*dp_table, 0)
                cam_t_obj = Transform(p=0.3 * dp_cam) @ cam_t_obj
            mx, my = x, y
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            if mx is not None and my is None:
                # TODO: rotate as if the mouse is dragging the object around the rotation axis
                phi = (x - mx) * np.pi / 360
                cam_t_obj = cam_t_obj @ table_t_obj_stable.inv @ Transform(rotvec=(0, 0, phi)) @ table_t_obj_stable
            mx, my = x, None
        else:
            mx, my = None, None
        draw()


    draw()
    cv2.setMouseCallback('', mouse_cb)
    while True:
        key = cv2.waitKey()
        if key == ord('q'):
            quit()
        elif key == ord(' '):
            cam_t_obj.save(annotation_path)
            break
        elif key == 8 or key == 255:  # Backspace, Delete
            annotation_path.open('w').close()
            break
        elif key == ord('r'):
            cam_t_obj = cam_t_obj_init
