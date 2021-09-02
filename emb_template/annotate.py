import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from transform3d import Transform
import tqdm

from . import vis
from . import utils
from . import camera
from .renderer import MeshRenderer

parser = argparse.ArgumentParser()
parser.add_argument('image_folder')
parser.add_argument('object_name')
parser.add_argument('--oldest-first', action='store_true')
parser.add_argument('--translation-scaling', type=float, default=0.3)
parser.add_argument('--rotation-scaling', type=float, default=0.008)
parser.add_argument('--render-color', type=float, nargs=3, default=(1., .4, .2))
args = parser.parse_args()

object_name = args.object_name
image_folder = Path(args.image_folder)
assert image_folder.is_dir()
obj_folder = Path('objects') / object_name
assert obj_folder.is_dir()
mesh = trimesh.load_mesh(obj_folder / 'cad.stl')
cam_info = camera.CameraInfo.load()
K = cam_info.K
annotation_folder = utils.get_annotation_folder(object_name, image_folder)
annotation_folder.mkdir(exist_ok=True, parents=True)

image_names = utils.load_image_names(image_folder)
annotation_names = utils.load_annotation_names(object_name, image_folder)
annotations_required = set(image_names) - set(annotation_names)
annotations_required = sorted(annotations_required, reverse=not args.oldest_first)
if not annotations_required:
    print(f'No images need "{object_name}" annotations')
    quit()

renderer = MeshRenderer(mesh=mesh, h=cam_info.h, w=cam_info.w, K=cam_info.K, color=args.render_color)
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
      '  - s to skip\n'
      '  - h to hide / unhide render\n'
      '  - q to quit')

for image_name in tqdm.tqdm(annotations_required):
    annotation_path = annotation_folder / f'{image_name}.txt'
    img = cv2.imread(str(image_folder / f'{image_name}.png'))
    prediction_path = image_folder / f'{image_name}.{object_name}.txt'
    if prediction_path.exists():
        cam_t_obj = Transform.load(prediction_path)
    else:
        cam_t_obj = cam_t_obj_init
    mx, my = None, None  # mouse position
    hidden = False


    def draw():
        global hidden
        render = renderer.render(cam_t_obj)[0]
        cv2.imshow('', vis.composite(img, render[..., :3], render[..., 3:] // 2))
        hidden = False


    def mouse_cb(event, x, y, flags, _):
        global mx, my, cam_t_obj, state
        if mx is None:
            pass
        elif flags & cv2.EVENT_FLAG_CTRLKEY:  # translate
            dp_img = np.array((x - mx, y - my))
            jac_img_xy = K[:2, :2] / cam_t_obj.p[2]
            jac_img_table = jac_img_xy @ cam_t_table.R[:2, :2]
            # jac_img_table @ dp_table = dp_img, solve for dp_table:
            dp_table = np.linalg.solve(jac_img_table, dp_img)
            dp_cam = cam_t_table.R @ (*dp_table, 0)
            cam_t_obj = Transform(p=args.translation_scaling * dp_cam) @ cam_t_obj
            draw()
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:  # rotate
            p_center = K @ cam_t_obj.p
            p_center = p_center[:2] / p_center[2]
            p_prev, p_now = (mx, my) - p_center, (x, y) - p_center
            phi = np.arctan2(*p_prev[::-1]) - np.arctan2(*p_now[::-1])
            cam_t_obj = cam_t_obj @ table_t_obj_stable.inv @ Transform(rotvec=(0, 0, phi)) @ table_t_obj_stable
            draw()
        mx, my = x, y


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
