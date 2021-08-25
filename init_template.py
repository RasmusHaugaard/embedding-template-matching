import argparse
import numpy as np
import cv2
import enum

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--image', required=True)
args = parser.parse_args()

img = cv2.imread(args.image)
mask = np.zeros(img.shape[:2], dtype=np.uint8)


class State(enum.Enum):
    SELECT_CENTER = 0
    PAINT_MASK = 1


state = State.SELECT_CENTER
c = (255, 255, 255)
center = None


def cb(event, x, y, flags, params):
    global state, center
    if state is State.SELECT_CENTER:
        temp = img.copy()
        cv2.drawMarker(temp, (x, y), c, cv2.MARKER_CROSS)
        if event == cv2.EVENT_LBUTTONUP:
            center = x, y
            state = State.PAINT_MASK
            temp = img
        cv2.imshow('', temp)
    elif state is State.PAINT_MASK:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                cv2.circle(mask, (x, y), 10, 0, -1)
            else:
                cv2.circle(mask, (x, y), 10, 1, -1)
        temp = img.copy()
        mask_ = mask == 1
        temp[mask_] = temp[mask_] // 2 + (0, 0, 127)
        cv2.imshow('', temp)
    else:
        raise ValueError()


cv2.imshow('', img)
cv2.setMouseCallback('', cb)
cv2.waitKey()

mask_args = np.argwhere(mask == 1)  # (N, 2)

x, y = center
k = int(np.ceil(np.linalg.norm((y, x) - mask_args, axis=-1).max()))

img_crop = img[y - k:y + k + 1, x - k:x + k + 1]
cv2.imshow('', img_crop)

mask_img = np.zeros(img_crop.shape[:2], dtype=np.uint8)
mask_args_centered = mask_args - (y, x) + k
yy, xx = mask_args_centered.T
mask_img[yy, xx] = 255
cv2.imshow('mask', mask_img)

img_crop = np.concatenate((img_crop, mask_img[..., None]), axis=-1)
cv2.imwrite(f'templates/{args.name}.png', img_crop)
