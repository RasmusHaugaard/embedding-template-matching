import cv2
import numpy as np

import vis


class Annotator:
    def __init__(self, im, rgba_template, cx=None, cy=None, theta=None):
        self.im = im
        self.h, self.w = im.shape[:2]
        self.rgba_template = rgba_template
        self.k = rgba_template.shape[0] // 2

        self.theta = 0 if theta is None else theta
        self.cx = self.w // 2 if cx is None else cx
        self.cy = self.h // 2 if cy is None else cy

        self.x, self.y = None, None

    def draw(self):
        temp = vis.overlay_template(self.im, self.rgba_template, self.cx, self.cy, self.theta)
        cv2.imshow('', temp)

    def mouse_cb(self, event, xx, yy, flags, _):
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if self.x is not None and self.y is not None:
                dx, dy = xx - self.x, yy - self.y
                self.cx += dx / 2
                self.cy += dy / 2
            self.x, self.y = xx, yy
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            if self.x is not None and self.y is None:
                self.theta += (xx - self.x) * np.pi / 360
            self.x, self.y = xx, None
        else:
            self.x, self.y = None, None
        self.draw()

    def run(self):
        self.draw()
        cv2.setMouseCallback('', self.mouse_cb)
        while True:
            key = cv2.waitKey()
            if key == ord('q'):
                return None
            elif key == ord(' '):
                return self.cx, self.cy, self.theta


def _main():
    import argparse
    from pathlib import Path

    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()

    rgba_template = utils.load_rgba_template(args.name)

    for fp in Path('images').glob('*.png'):
        img = cv2.imread(str(fp))
        i = int(fp.name.split('.')[0])
        fp_anno = Path(f'annotations/{i}.{args.name}.txt')
        if fp_anno.exists():
            cx, cy, theta = np.loadtxt(str(fp_anno))
        else:
            cx, cy, theta = None, None, None
        anno = Annotator(im=img, rgba_template=rgba_template, cx=cx, cy=cy, theta=theta).run()
        if anno is None:
            break
        cx, cy, theta = anno
        np.savetxt(str(fp_anno), (cx, cy, theta))


if __name__ == '__main__':
    _main()
