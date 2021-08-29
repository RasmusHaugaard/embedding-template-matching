import json
from pathlib import Path
import argparse

import rospy
import sensor_msgs.msg
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('image_topic')
parser.add_argument('--overwrite', '-y', action='store_true')
args = parser.parse_args()

scene_fp = Path('cam_info.json')
if scene_fp.exists() and not args.overwrite:
    raise RuntimeError('camera info is already initialized')

rospy.init_node('init_cam_info', anonymous=True)

msg = rospy.wait_for_message(f'{args.image_topic}/camera_info', sensor_msgs.msg.CameraInfo)
D = np.array(msg.D)
K = np.array(msg.K).reshape(3, 3)
h, w = msg.height, msg.width

assert np.all(D == 0) and np.all(K[(0, 1), (1, 0)] == 0), 'camera must be rectified'

Path('images').mkdir(exist_ok=args.overwrite)

json.dump(dict(
    image_topic=args.image_topic,
    K=K.tolist(),
    h=h, w=w
), scene_fp.open('w'), indent=2)

print('camera intiialized')
