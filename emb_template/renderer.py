import os

import numpy as np
import trimesh
from transform3d import Transform

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

opengl_t_opencv = Transform(rotvec=(np.pi, 0, 0))


class MeshRenderer:
    def __init__(self, mesh: trimesh.Trimesh, h: int, w: int, K: np.ndarray = None, yfov=np.deg2rad(45)):
        self.h, self.w = h, w
        self.center = mesh.bounding_sphere.primitive.center
        self.diameter = mesh.bounding_sphere.primitive.radius * 2
        self.scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=0.1)
        self.scene.add(pyrender.DirectionalLight(color=(1., 1., 1.), intensity=5), pose=opengl_t_opencv.matrix)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        znear = self.diameter / 10
        if K is not None:
            assert np.all(K[(0, 1), (1, 0)] == 0), 'can only visualize poses from a rectified image'
            cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=znear)
        else:
            cam = pyrender.PerspectiveCamera(yfov=yfov, znear=znear, aspectRatio=w / h)
        self.cam_node = self.scene.add(cam, pose=opengl_t_opencv.matrix)
        self.obj_node = self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    def render(self, cam_t_obj: Transform):
        self.scene.set_pose(self.obj_node, cam_t_obj.matrix)
        img, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return img, depth


def _main():
    import cv2
    mesh_renderer = MeshRenderer(trimesh.primitives.Sphere(), 512, 512)
    img, _ = mesh_renderer.render(cam_t_obj=Transform(p=(0, 0, 3)))
    cv2.imshow('', img)
    cv2.waitKey()


if __name__ == '__main__':
    _main()
