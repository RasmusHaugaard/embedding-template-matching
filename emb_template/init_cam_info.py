import json
from pathlib import Path
import argparse

import rospy
import sensor_msgs.msg
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--image-topic', required=True)
parser.add_argument('--overwrite', '-y', action='store_true')
args = parser.parse_args()

scene_fp = Path('cam_info.json')
if scene_fp.exists() and not args.overwrite:
    raise RuntimeError('camera info is already initialized')

rospy.init_node('init_cam_info', anonymous=True)

msg = rospy.wait_for_message(f'{args.image_topic}/camera_info', sensor_msgs.msg.CameraInfo)
D = np.array(msg.D)
K = np.array(msg.K).reshape(3, 3)

Path('images').mkdir(exist_ok=args.overwrite)
Path('rgba_templates').mkdir(exist_ok=args.overwrite)
Path('annotations').mkdir(exist_ok=args.overwrite)
Path('models').mkdir(exist_ok=args.overwrite)

json.dump(dict(
    image_topic=args.image_topic,
    K=K.tolist(),
    D=D.tolist(),
), scene_fp.open('w'), indent=2)
