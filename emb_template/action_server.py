import datetime
import argparse
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
import rospy
import actionlib
import sensor_msgs.msg
import geometry_msgs.msg
from transform3d import Transform

import emb_template_ros.msg
from .camera import Camera, CameraInfo
from . import utils
from . import vis
from .renderer import MeshRenderer
from .model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='emb_template')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
name = args.name
device = torch.device(args.device)

rospy.init_node(name, anonymous=True)
pub_image_annotated = rospy.Publisher(f'{name}/pose_visualized', sensor_msgs.msg.Image, queue_size=1)

cam_info = CameraInfo.load()
cam = Camera(cam_info.image_topic)
cam_t_table = Transform.load('cam_t_table.txt')

log_folder = Path('action_server_log')
log_folder.mkdir(exist_ok=True)

gpu_lock = threading.Lock()


@torch.no_grad()
def execute_cb(goal):
    object_name = goal.object_name
    rgba_template, table_offset, obj_t_template, sym = utils.load_current_template(object_name)

    with gpu_lock:
        model = Model.load_from_checkpoint(
            utils.latest_checkpoint(object_name),
            rgba_template=rgba_template, sym=sym,
        )
        model.eval()
        model.to(device)
        img = cam.take_image()
        act, _ = model.forward(utils.normalize(img)[None].to(device))
        stride = model.stride
        del model
        act = act[0].cpu().numpy()

        pose_2d = utils.pose_2d_from_act(act=act, stride=stride, sym=sym)
        cam_t_obj = utils.get_pose_3d(pose_2d=pose_2d, K=cam_info.K, cam_t_table=cam_t_table,
                                      table_offset=table_offset, obj_t_template=obj_t_template)

    result = emb_template_ros.msg.getPoseResult()
    result.pose = geometry_msgs.msg.Pose(
        position=geometry_msgs.msg.Point(*cam_t_obj.p),
        orientation=geometry_msgs.msg.Quaternion(*cam_t_obj.quat)
    )
    server.set_succeeded(result=result)

    # logging and debugging
    with gpu_lock:
        render = MeshRenderer(
            mesh=utils.load_mesh(object_name), h=cam_info.h, w=cam_info.w, K=cam_info.K,
        ).render(cam_t_obj)[0].copy()
    render[..., :2] = 0
    img_overlay = vis.composite(img, render[..., :3], render[..., 3:] // 2)  # type: np.ndarray
    img_msg = sensor_msgs.msg.Image(encoding='bgr8')
    contig = np.ascontiguousarray(img_overlay)
    img_msg.step = contig.strides[0]
    img_msg.data = contig.tobytes()
    img_msg.height, img_msg.width = img_overlay.shape[:2]
    pub_image_annotated.publish(img_msg)

    now = datetime.datetime.now()
    cv2.imwrite(str(log_folder / f'{now}.png'), img)
    cam_t_obj.save(log_folder / f'{now}.{object_name}.txt')


server = actionlib.SimpleActionServer(
    f'{args.name}/getPose',
    emb_template_ros.msg.getPoseAction,
    execute_cb=execute_cb,
    auto_start=False,
)
server.start()
rospy.spin()
