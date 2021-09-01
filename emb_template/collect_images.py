from pathlib import Path
import argparse
import datetime

import cv2
import rospy

from . import camera

parser = argparse.ArgumentParser()
parser.add_argument('image_folder', default='images')
args = parser.parse_args()

image_folder = Path(args.image_folder)
image_folder.mkdir(exist_ok=True)
rospy.init_node('collect_images', anonymous=True)

cam = camera.Camera()
i = 0
while True:
    img = cam.take_image()
    cv2.imshow('', img)
    now = datetime.datetime.now()
    key = cv2.waitKey(1)
    if key == ord(' '):
        cv2.imwrite(str(image_folder / f'{datetime.datetime.now()}.png'), img)
        i += 1
        print(f'{i} new image(s) taken')
    elif key == ord('q'):
        break
