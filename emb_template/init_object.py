import json
from pathlib import Path
import argparse

import cv2
import trimesh
from transform3d import Transform
import rospy

from . import vis
from . import utils
from . import camera
from .renderer import MeshRenderer

parser = argparse.ArgumentParser()
parser.add_argument('object_name')
parser.add_argument('cad')
parser.add_argument('--cad-scale', type=float, default=1e-3)
parser.add_argument('--image')
args = parser.parse_args()
object_name = args.object_name

# TODO: validate object name

obj_folder = Path('objects') / object_name
if obj_folder.exists():
    print('Object with that name already exists')
    quit()

mesh = trimesh.load_mesh(args.cad)  # type: trimesh.Trimesh
mesh.apply_scale(args.cad_scale)

cam_info = camera.CameraInfo.load()
K = cam_info.K
if args.image is not None:
    assert Path(args.image).is_file()
    img = cv2.imread(args.image)
else:
    rospy.init_node('init_object', anonymous=True)
    cam = camera.Camera(cam_info.image_topic)
    img = cam.take_image()

cam_t_table = Transform.load('cam_t_table.txt')
cam_t_table_pos = utils.get_cam_t_table_center(cam_t_table=cam_t_table, K=cam_info.K, w=cam_info.w, h=cam_info.h)
renderer = MeshRenderer(mesh=mesh, h=cam_info.h, w=cam_info.w, K=K)

print('Choose stable pose. '
      '  - Use "a" and "d" to switch between stable poses\n'
      '  - Confirm with Enter\n'
      '  - Abort with "q"')
stable_poses, pose_probs = mesh.compute_stable_poses()
i = 0


def draw():
    table_t_obj = Transform(matrix=stable_poses[i])
    cam_t_obj = cam_t_table_pos @ table_t_obj
    overlay = renderer.render(cam_t_obj)[0].copy()
    overlay[..., :2] = 0
    comp = vis.composite(img, overlay[..., :3], overlay[..., 3:] // 2)
    cv2.imshow('', comp)


def mouse_cb(event, x, y, flags, _):
    global cam_t_table_pos
    cam_t_table_pos = utils.get_cam_t_plane_ray(cam_t_plane=cam_t_table, K=cam_info.K, x=x, y=y)
    draw()


cv2.imshow('', img)
cv2.setMouseCallback('', mouse_cb)
while True:
    key = cv2.waitKey()
    if key == ord('q'):
        quit()
    elif key == ord('a'):
        i = max(i - 1, 0)
    elif key == ord('d'):
        i = min(i + 1, len(stable_poses) - 1)
    elif key == ord('\r'):
        break
    draw()
table_t_obj_stable = Transform(matrix=stable_poses[i])

obj_folder.mkdir(parents=True)
mesh.export(obj_folder / 'cad.stl')
table_t_obj_stable.save(obj_folder / 'table_t_obj_stable.txt')

print()
print(f'"{object_name}" initialized')
