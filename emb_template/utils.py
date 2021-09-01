from pathlib import Path
import datetime

import cv2
import numpy as np
import natsort
import albumentations as A
import trimesh
from albumentations.pytorch import ToTensorV2
from transform3d import Transform

_normalize = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


def normalize(img):
    return _normalize(image=img)['image']


def sorted_paths(paths):
    return natsort.natsorted(paths, key=lambda path: str(path))


def latest_checkpoint(object_name):
    models_folder = get_current_template_folder(object_name) / 'models'
    fp = sorted_paths(models_folder.glob('version_*'))[-1]
    fp = sorted_paths((fp / 'checkpoints').glob('*.ckpt'))[-1]
    return fp


def pose_2d_from_act(act: np.ndarray, stride: int, sym: int):
    theta, y, x = np.unravel_index(np.argmax(act), act.shape)
    y, x = y * stride, x * stride
    theta = 0 if sym == -1 else theta / len(act) * 2 * np.pi / sym
    return x, y, theta


def get_pose_3d(pose_2d, K: np.ndarray, cam_t_table: Transform,
                table_offset: float, obj_t_template: Transform):
    x, y, theta = pose_2d
    template_plane = cam_t_table @ Transform(p=(0, 0, table_offset))
    cam_t_template = get_cam_t_plane_ray(cam_t_plane=template_plane, K=K, x=x, y=y) @ Transform(rotvec=(0, 0, -theta))
    return cam_t_template @ obj_t_template.inv


def get_cam_t_plane_ray(cam_t_plane: Transform, K: np.ndarray, x: int, y: int):
    # TODO: documentation
    table_normal = cam_t_plane.R[:, 2]
    table_point = cam_t_plane.p
    ray_direction = np.linalg.inv(K) @ (x, y, 1)
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    ray_point = np.zeros(3)
    ww = ray_point - table_point
    si = -table_normal @ ww / (table_normal @ ray_direction)
    cam_p_plane_ray = ww + si * ray_direction + table_point
    cam_t_plane_ray = Transform(p=cam_p_plane_ray, R=cam_t_plane.R)
    return cam_t_plane_ray


def get_cam_t_table_center(cam_t_table: Transform, K: np.ndarray, w: int, h: int):
    return get_cam_t_plane_ray(cam_t_plane=cam_t_table, K=K, x=w // 2, y=h // 2)


def log_prediction(object_name: str, img: np.ndarray, cam_t_obj: Transform):
    log_folder = Path('log') / object_name
    log_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.datetime.now()
    cv2.imwrite(str(log_folder / f'{now}.png'), img)
    cam_t_obj.save(log_folder / f'{now}.{object_name}.txt')


def get_object_folder(object_name):
    return Path('objects') / object_name


def load_mesh(object_name):
    return trimesh.load_mesh(get_object_folder(object_name) / 'cad.stl')


def get_annotation_folder(object_name, image_folder):
    return get_object_folder(object_name) / 'annotations' / image_folder


def get_templates_folder(object_name):
    return get_object_folder(object_name) / 'templates'


def get_template_folder(object_name, template_name):
    return get_templates_folder(object_name) / template_name


def get_current_template_name(object_name):
    return (get_object_folder(object_name) / 'current_template.txt').open().read().strip()


def get_current_template_folder(object_name):
    return get_template_folder(object_name, get_current_template_name(object_name))


def load_image_names(image_folder):
    return sorted([fp.name[:-4] for fp in Path(image_folder).glob('*.png')])


def load_annotation_names(object_name, image_folder):
    annotation_folder = get_annotation_folder(object_name, image_folder)
    assert annotation_folder.exists()
    return sorted([fp.name[:-4] for fp in annotation_folder.glob('*.txt')])


def load_annotation(fp):
    with fp.open() as f:
        text = f.read()
    if text.strip() == '':
        return None
    else:
        return Transform.load(fp)


def load_valid_annotation_and_image_fps(object_name):
    annotation_folder = get_annotation_folder(object_name, '')
    annotation_fps = []
    image_fps = []
    for fp in annotation_folder.glob('**/*.txt'):
        annotation = load_annotation(fp)
        if annotation is not None:
            annotation_fps.append(fp)
            relative_dir = fp.parent.relative_to(annotation_folder)
            image_fps.append(relative_dir / f'{fp.name[:-4]}.png')
    return annotation_fps, image_fps


def resize(img: np.ndarray, img_scale: float, interp=cv2.INTER_LINEAR):
    h, w = (int(np.floor(r * img_scale)) for r in img.shape[:2])
    M = np.eye(3) * img_scale
    M[2, 2] = 1.
    img = cv2.warpAffine(img, M[:2], (w, h), flags=interp)
    return img, M


def load_template(object_name, template_name):
    template_folder = get_template_folder(object_name, template_name)
    rgba_template = cv2.imread(str(template_folder / 'rgba_template.png'), cv2.IMREAD_UNCHANGED)
    obj_t_template = Transform.load(template_folder / 'obj_t_template.txt')
    sym = int((template_folder / 'sym.txt').open().read())
    table_offset = float((template_folder / 'table_offset.txt').open().read())
    return rgba_template, table_offset, obj_t_template, sym


def load_current_template(object_name):
    return load_template(object_name, get_current_template_name(object_name))
