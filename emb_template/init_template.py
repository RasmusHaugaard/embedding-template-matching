import argparse

import cv2
import numpy as np
from transform3d import Transform

from . import vis
from . import utils
from . import camera
from .renderer import MeshRenderer

# UX:
#   choose image from labeled images
#   choose table height
#   choose template center
#   draw mask
#   choose sym


parser = argparse.ArgumentParser()
parser.add_argument('object_name')
parser.add_argument('template_name')
args = parser.parse_args()
object_name = args.object_name
template_name = args.template_name

template_folder = utils.get_template_folder(object_name=object_name, template_name=template_name)
if template_folder.exists():
    print('Template with that name already exists.')
    quit()

annotation_fps, image_fps = utils.load_valid_annotation_and_image_fps(object_name)
if not annotation_fps:
    print('Annotate some poses first.')
    quit()

cam_info = camera.CameraInfo.load()
cam_t_table = Transform.load('cam_t_table.txt')
mesh = utils.load_mesh(object_name)
renderer = MeshRenderer(mesh=mesh, K=cam_info.K, h=cam_info.h, w=cam_info.w)

print()
print('Use "a" and "d" to select an image for template initialization.\n'
      'Confirm with Enter.')

i = 0
while True:
    img = cv2.imread(str(image_fps[i]))
    print(image_fps[i])
    cam_t_obj = Transform.load(annotation_fps[i])
    overlay, depth = renderer.render(cam_t_obj)
    comp = vis.composite(img, overlay[..., :3], overlay[..., 3:])
    cv2.imshow('', comp)
    key = cv2.waitKey()
    if key == ord('q'):
        quit()
    elif key == ord('a'):
        i = (i - 1) % len(annotation_fps)
    elif key == ord('d'):
        i = (i + 1) % len(annotation_fps)
    elif key == ord('\r'):
        break

print()
print('Choose an offset from the table plane to the template matching plane\n'
      'by choosing a point on that plane on the object render.\n'
      'Set offset to zero by clicking anywhere else.\n'
      'Press "m" to enter the offset manually.\n'
      'Confirm with Enter.')

table_offset = 0.


def cb(event, x_img, y_img, flags, _):
    global table_offset
    if event == cv2.EVENT_LBUTTONUP:
        z = depth[y_img, x_img]
        if z == 0:
            table_offset = 0.
        else:
            line_cam = np.linalg.inv(cam_info.K) @ (x_img, y_img, 1)
            pt_cam = line_cam / line_cam[2] * depth[y_img, x_img]
            pt_table = cam_t_table.inv @ pt_cam
            table_offset = pt_table[2]
        print(f'Offset: {table_offset:.4f} m')


cv2.setMouseCallback('', cb)
while True:
    key = cv2.waitKey()
    if key == ord('q'):
        quit()
    elif key == ord('\r'):
        break
    elif key == ord('m'):
        table_offset = float(input('Enter table offset in meters:'))
        break

print()
print('Choose the template center. Confirm with Enter.')
template_center = None


def cb(event, x_img, y_img, flags, _):
    global template_center
    if flags & cv2.EVENT_FLAG_LBUTTON:
        template_center = x_img, y_img
        comp_temp = comp.copy()
        cv2.drawMarker(comp_temp, (x_img, y_img), (0, 0, 255), cv2.MARKER_CROSS, 30)
        cv2.imshow('', comp_temp)


cv2.setMouseCallback('', cb)
while True:
    key = cv2.waitKey()
    if key == ord('q'):
        quit()
    elif key == ord('\r') and template_center is not None:
        break

print()
print('Draw template mask. Confirm with Enter.')
mask = np.zeros(img.shape[:2], dtype=np.uint8)
radius = 8
mx, my = 0, 0


def draw():
    temp = img.copy()
    cv2.drawMarker(temp, template_center, (0, 0, 255), cv2.MARKER_CROSS, 30)
    mask_ = mask == 255
    temp[mask_] = temp[mask_] // 2 + (0, 0, 127)
    mouse_mask = np.zeros_like(mask)
    cv2.circle(mouse_mask, (mx, my), radius, 255, -1)
    mouse_mask = mouse_mask == 255
    temp[mouse_mask] = temp[mouse_mask] // 2 + (127, 0, 0)
    cv2.imshow('', temp)


def cb(event, x, y, flags, _):
    global mx, my
    mx, my = x, y
    if flags & cv2.EVENT_FLAG_LBUTTON:
        color = 0 if flags & cv2.EVENT_FLAG_CTRLKEY else 255
        cv2.circle(mask, (x, y), radius, color, -1)
    draw()


draw()
cv2.setMouseCallback('', cb)
while True:
    key = cv2.waitKey()
    if key == ord('q'):
        quit()
    if key == ord('a'):
        radius = max(1, radius // 2)
    elif key == ord('d'):
        radius = radius * 2
    elif key == ord('\r'):
        break
    draw()

# sym
while True:
    print()
    sym = input('Write the number of discrete rotational symmetries in the terminal:\n'
                '  - 1 for no symmetry\n'
                '  - 2 for 180° symmetry\n'
                '  - 3 for 120° symmetry\n'
                '  - ...\n'
                '  - -1 for full rotational symmetry\n')
    try:
        sym = int(sym)
        break
    except ValueError:
        print('Could not convert input to integer. Try again.')

# save
# template folder
template_folder.mkdir(parents=True)

# obj_t_template
cam_t_template_plane = cam_t_table @ Transform(p=(0, 0, table_offset))
cam_t_template = utils.get_cam_t_plane_ray(
    cam_t_plane=cam_t_template_plane, K=cam_info.K, x=template_center[0], y=template_center[1],
)
obj_t_template = cam_t_obj.inv @ cam_t_template
obj_t_template.save(template_folder / 'obj_t_template.txt')

# table offset
with (template_folder / 'table_offset.txt').open('w') as f:
    f.write(str(table_offset))

# mask
mask_args = np.argwhere(mask == 255)  # (N, 2)
mx, my = template_center
k = int(np.ceil(np.linalg.norm((my, mx) - mask_args, axis=-1).max()))
rgba_template = np.concatenate((
    img[my - k:my + k + 1, mx - k:mx + k + 1],
    mask[my - k:my + k + 1, mx - k:mx + k + 1, None],
), axis=-1)
cv2.imwrite(str(template_folder / 'rgba_template.png'), rgba_template)

# sym
with (template_folder / 'sym.txt').open('w') as f:
    f.write(str(sym))

# make current template
with (utils.get_object_folder(object_name) / 'current_template.txt').open('w') as f:
    f.write(template_name)

print()
print('Template initialized and made the current template.')
