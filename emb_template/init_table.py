import argparse

import numpy as np
import cv2
import rospy
from transform3d import Transform

from . import camera

parser = argparse.ArgumentParser()
parser.add_argument('--from-file')
parser.add_argument('--dictionary', default='4X4_100')
parser.add_argument('--marker-length', type=float, default=0.015)
parser.add_argument('--square-length', type=float, default=0.020)
parser.add_argument('--squares-x', type=int, default=14)
parser.add_argument('--squares-y', type=int, default=9)
parser.add_argument('--board-height', type=float, default=0.003)
parser.add_argument('--no-debug', action='store_true')

args = parser.parse_args()

dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f'DICT_{args.dictionary.upper()}'))
board = cv2.aruco.CharucoBoard_create(
    squaresX=args.squares_x, squaresY=args.squares_y,
    squareLength=args.square_length, markerLength=args.marker_length, dictionary=dictionary
)  # type: cv2.aruco_Board

rospy.init_node('init_table', anonymous=True)
cam_info = camera.CameraInfo.load()

if args.from_file is not None:
    img = cv2.imread(args.from_file)
else:
    cam = camera.Camera(cam_info.image_topic)
    img = cam.take_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners, ids, _ = cv2.aruco.detectMarkers(
    image=gray, dictionary=dictionary, cameraMatrix=cam_info.K, distCoeff=None
)
if not corners:
    print('did not find charuco markers')
    cv2.imshow('', img)
    cv2.waitKey()
    quit()

ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
    markerCorners=corners, markerIds=ids, image=gray, board=board
)
assert ret

rvec, tvec = np.zeros(3), np.zeros(3)
cv2.aruco.estimatePoseCharucoBoard(
    charucoCorners=np.array(charuco_corners), charucoIds=np.array(charuco_ids), board=board,
    cameraMatrix=cam_info.K, distCoeffs=None,
    rvec=rvec, tvec=tvec
)

cam_t_board = Transform(p=tvec, rotvec=rvec)
angle = np.arccos(cam_t_board.R[:, 2] @ (0, 0, -1))
print('This works best when the table is parallel to the image plane.')
print(f'Table is {np.rad2deg(angle):.1f} degrees off with respect to the image plane')

if not args.no_debug:
    img_detections = img.copy()
    cv2.aruco.drawDetectedCornersCharuco(image=img_detections, charucoCorners=charuco_corners)
    cv2.drawFrameAxes(
        image=img_detections, cameraMatrix=cam_info.K, distCoeffs=None,
        rvec=rvec, tvec=tvec, length=args.squares_y * args.square_length,
    )
    cv2.imshow('', img_detections)
    print('Press q to abort')
    if cv2.waitKey() == ord('q'):
        quit()

cam_t_board = Transform(p=tvec, rotvec=rvec)
cam_t_table = cam_t_board @ Transform(p=(0, 0, -args.board_height))
z = cam_t_table.R[:, 2]
y = np.cross(z, (1, 0, 0))
y = y / np.linalg.norm(y)
x = np.cross(y, z)
R = np.stack((x, y, z), axis=1)
cam_t_table = Transform(R=R, p=cam_t_table.p)
cam_t_table.save('cam_t_table.txt')

print('Table is initialized')