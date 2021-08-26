import time
import numpy as np
import pyrealsense2 as rs


class Camera:
    def __init__(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipe.start(cfg)  # type: rs.pipeline_profile
        self.device = self.profile.get_device()  # type: rs.device
        self.set_roi(1060, 560, 1180, 680)

    def set_roi(self, l, t, r, b, n_tries=100, try_delay=0.01):
        s = self.device.first_color_sensor().as_roi_sensor()
        roi = s.get_region_of_interest()  # type: rs.region_of_interest
        roi.min_x, roi.min_y = l, t
        roi.max_x, roi.max_y = r, b
        success = False
        for i in range(n_tries):
            try:
                s.set_region_of_interest(roi)
                success = True
                break
            except RuntimeError:
                time.sleep(try_delay)
        if not success:
            raise RuntimeError()

    def take_image(self):
        frame = self.pipe.wait_for_frames().get_color_frame()
        img = np.asanyarray(frame.get_data())[:704][::-1, ::-1].copy()
        return img


def _main():
    from pathlib import Path
    import cv2

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


if __name__ == '__main__':
    _main()
