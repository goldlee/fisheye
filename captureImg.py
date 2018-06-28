#--coding:utf-8
import cv2
import numpy
import os

cap = cv2.VideoCapture(0)
CHECKERBOARD = (6, 9)
save_dir = './ciga_pic/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
n = 0
while(1):
    ret, frame = cap.read()
    image = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        showImg = cv2.drawChessboardCorners(frame, (6, 9), corners, ret)

        cv2.imshow("show", showImg)
        c = cv2.waitKey(1000)
        if c == ord('s'):
            cv2.imwrite(os.path.join(save_dir, str(n) + '.jpg'), image)
            n += 1
            print("images:",n)

    cv2.imshow("show", frame)
    k = cv2.waitKey(200)
    if k == ord('i'):
        cv2.imwrite(os.path.join(save_dir, str(n) + '.jpg'), frame)
        n += 1
        print("images:", n)
    if k == ord('q'):
        print("bbb")
        break