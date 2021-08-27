from pathlib import Path

import cv2

from . import camera
from . import utils

cam_info = utils.load_cam_info()

img_fps = list(Path('images').glob('*.png'))
if not img_fps:
    i = 0
else:
    i = max([int(fp.name.split('.')[0]) for fp in img_fps])

cam = Camera()
while True:
    img = cam.take_image()
    cv2.imshow('', img)
    key = cv2.waitKey(1)
    if key == ord(' '):
        cv2.imwrite(f'images/{i}.png', img)
        i += 1
        print(i)
    elif key == ord('q'):
        break
