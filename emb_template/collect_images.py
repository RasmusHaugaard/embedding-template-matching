import datetime

import cv2
import rospy

from . import camera

rospy.init_node('collect_images', anonymous=True)

cam = camera.Camera()
i = 0
while True:
    img = cam.take_image()
    cv2.imshow('', img)
    now = datetime.datetime.now()
    key = cv2.waitKey(1)
    if key == ord(' '):
        cv2.imwrite(f'images/{datetime.datetime.now()}.png', img)
        i += 1
        print(f'{i} new image(s) taken')
    elif key == ord('q'):
        break
