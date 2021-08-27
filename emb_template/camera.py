from typing import Union
import time
import json

import numpy as np
import rospy
import sensor_msgs.msg


class Camera:
    def __init__(self, image_topic: str = None):
        if image_topic is None:
            image_topic = CameraInfo.load().image_topic
        rospy.Subscriber(f'{image_topic}/image_raw', sensor_msgs.msg.Image, self._temp_store_img, queue_size=1)
        self.last_msg = None  # type: Union[None, sensor_msgs.msg.Image]

    def _temp_store_img(self, msg: sensor_msgs.msg.Image):
        self.last_msg = msg

    def take_image(self, timeout=2.):
        start = time.time()
        img = None
        while time.time() - start < timeout:
            if self.last_msg is not None:
                msg = self.last_msg
                self.last_msg = None
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[..., ::-1]
                h, w = img.shape[:2]
                # must be multiple of 32 (requirement of unet model)
                h, w = h // 32 * 32, w // 32 * 32
                img = img[:h, :w].copy()  # copy releases memory
                break
            else:
                time.sleep(0.002)
        if img is None:
            raise TimeoutError()
        return img


class CameraInfo:
    def __init__(self, image_topic: str, K: np.ndarray, D: np.ndarray):
        self.image_topic = image_topic
        self.K = K
        self.D = D

    @classmethod
    def load(cls):
        d = json.load(open('cam_info.json'))
        return cls(
            image_topic=d['image_topic'],
            K=np.array(d['K']),
            D=np.array(d['D']),
        )


def _main():
    import cv2

    rospy.init_node('camera', anonymous=True)
    cam = Camera()
    while True:
        img = cam.take_image()
        cv2.imshow('', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    _main()
