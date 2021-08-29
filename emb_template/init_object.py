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
args = parser.parse_args()
object_name = args.object_name

obj_folder = Path('objects') / object_name
if obj_folder.exists():
    print('Object with that name already exists')
    quit()

mesh = trimesh.load_mesh(args.cad)  # type: trimesh.Trimesh
mesh.apply_scale(args.cad_scale)

rospy.init_node('init_object', anonymous=True)

cam_info = camera.CameraInfo.load()
K = cam_info.K
cam = camera.Camera(cam_info.image_topic)
img = cam.take_image()

cam_t_table = Transform.load('cam_t_table.txt')
cam_t_table_center = utils.get_cam_t_table_center(cam_t_table=cam_t_table, K=cam_info.K, w=cam_info.w, h=cam_info.h)
renderer = MeshRenderer(mesh=mesh, h=cam_info.h, w=cam_info.w, K=K)

# TODO: make it easier to evaluate a pose
#  maybe by moving the object with the mouse, or rendering top / left / front views

print('Choose stable pose. Use "a" and "d" to switch between stable poses. End with enter.')
stable_poses, pose_probs = mesh.compute_stable_poses()
i = 0
while True:
    table_t_obj = Transform(matrix=stable_poses[i])
    cam_t_obj = cam_t_table_center @ table_t_obj
    overlay, depth = renderer.render(cam_t_obj)
    comp = vis.composite(img, overlay[..., :3], overlay[..., 3:] // 3 * 2)
    cv2.imshow('', comp)
    key = cv2.waitKey()
    if key == ord('q'):
        quit()
    elif key == ord('a'):
        i = max(i - 1, 0)
    elif key == ord('d'):
        i = min(i + 1, len(stable_poses) - 1)
    elif key == ord('\r'):
        break
table_t_obj_stable = Transform(matrix=stable_poses[i])

obj_folder.mkdir(parents=True)
mesh.export(obj_folder / 'cad.stl')
table_t_obj_stable.save(obj_folder / 'table_t_obj_stable.txt')
json.dump(dict(
    from_cad=True,
), (obj_folder / 'config.json').open('w'), indent=2)

print()
print(f'"{object_name}" initialized')
