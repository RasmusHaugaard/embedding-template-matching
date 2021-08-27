import json
from pathlib import Path
import argparse

import numpy as np
import cv2
import trimesh
from transform3d import Transform
import rospy

from . import vis
from . import camera
from .renderer import MeshRenderer

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--mesh', required=True)
parser.add_argument('--mesh-scale', type=float, default=1e-3)
args = parser.parse_args()

obj_folder = Path('objects') / args.name
if obj_folder.exists():
    print('Object with that name already exists')
    quit()

mesh = trimesh.load_mesh(args.mesh)  # type: trimesh.Trimesh
mesh.apply_scale(args.mesh_scale)

rospy.init_node('init_object', anonymous=True)

cam_info = camera.CameraInfo.load()
K = cam_info.K
cam = camera.Camera(cam_info.image_topic)
img = cam.take_image()

cam_t_table = Transform.load('cam_t_table.txt')
renderer = MeshRenderer(mesh=mesh, h=cam_info.h, w=cam_info.w, K=K)

print('Choose stable pose. Use "w" and "e" to switch between stable poses. End with enter.')
stable_poses, pose_probs = mesh.compute_stable_poses()
i = 0
while True:
    table_t_obj = Transform(matrix=stable_poses[i])
    cam_t_obj = cam_t_table @ table_t_obj
    overlay, depth = renderer.render(cam_t_obj)
    comp = vis.composite(img, overlay[..., :3], overlay[..., 3:] // 3 * 2)
    cv2.imshow('', comp)
    key = cv2.waitKey()
    if key == ord('q'):
        quit()
    elif key == ord('w'):
        i = max(i - 1, 0)
    elif key == ord('e'):
        i = min(i + 1, len(stable_poses) - 1)
    elif key == ord('\r'):
        break
table_t_obj_stable = Transform(matrix=stable_poses[i])
print()

print('Choose an offset from the table plane to the template matching plane\n'
      'by choosing a point on that plane on the object render.\n'
      'Set offset to zero by clicking anywhere else.\n'
      'Confirm with Enter.')
table_template_offset = 0.


def cb(event, x_img, y_img, flags, _):
    global table_template_offset
    if event == cv2.EVENT_LBUTTONUP:
        z = depth[y_img, x_img]
        if z == 0:
            table_template_offset = 0.
        else:
            line_cam = np.linalg.inv(K) @ (x_img, y_img, 1)
            pt_cam = line_cam / line_cam[2] * depth[y_img, x_img]
            pt_table = cam_t_table.inv @ pt_cam
            table_template_offset = pt_table[2]
        print(f'Offset: {table_template_offset:.4f} m')


cv2.setMouseCallback('', cb)
while True:
    key = cv2.waitKey()
    if key == ord('q'):
        quit()
    elif key == ord('\r'):
        break

obj_folder.mkdir(parents=True)
mesh.export(obj_folder / 'mesh.stl')
table_t_obj_stable.save(obj_folder / 'table_t_obj_stable.txt')
json.dump(dict(
    from_cad=True,
    table_template_offset=table_template_offset,
), (obj_folder / 'config.json').open('w'), indent=2)

print(f'"{args.name}" initialized')

# TODO: with CAD, actual poses are annotated, and the template plane can be changed while keeping the annotations,
#  so detach template plane from stable pose definition.
