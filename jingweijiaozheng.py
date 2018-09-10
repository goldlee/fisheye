import cv2
import numpy as np
import glob
import math
import random
import time
import datetime


def GetMinCircle(img, scale=4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # 这里缩小了，所以后面的要乘scale
    th = cv2.resize(th, (th.shape[1] // scale, th.shape[0] // scale))
    im, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    # print(cnt.shape)
    centers, radius = cv2.minEnclosingCircle(cnt)
    radius = radius * 1
    cv2.circle(img, (int(centers[0] * scale), int(centers[1] * scale)), int(radius * scale), (255, 0, 0), 3)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)

    return (centers[0] * scale, centers[1] * scale), radius * scale


def fisheye_calib_normal(img, center_x, center_y, radius, hor_l_ang=40, hor_r_ang=140, ver_t_ang=50,ver_b_ang=130,dst_size=400):
    height, width = img.shape[0:2]
    off_x = width / 2 - radius
    off_y = height / 2 - radius
    testImg = np.zeros((height, width), np.uint8)
    dstImg = np.zeros((dst_size, dst_size, 3), np.uint8)
    for i in range(dst_size):
        for j in range(dst_size):
            alpha = (j + 1) / dst_size * (hor_r_ang - hor_l_ang) + hor_l_ang
            alpha = (alpha - 90) / 90
            beta = (i + 1) / dst_size * (ver_b_ang - ver_t_ang) + ver_t_ang
            beta = (beta - 90) / 90

            a = alpha * alpha
            b = beta * beta
            c = a * b
            d = a - c
            e = b - c
            c = 1 - c

            x = math.sqrt(d / c)
            y = math.sqrt(e / c)

            x = x if alpha > 0 else -x
            y = y if beta > 0 else -y

            x = radius * (x + 1) + off_x
            y = radius * (y + 1) + off_y

            x = int(x) if x < width else width-1
            y = int(y) if y < height else height-1

            # print(y, x)
            dstImg[i][j] = img[y][x]
            testImg[y][x] = 255

            # if i == 0 or i == dst_size-1:
            #     img[y][x] = (0, 255, 0)
            # if j == 0 or j == dst_size-1:
            #     img[y][x] = (0, 255, 0)

    # cv2.imshow('test', img)
    # cv2.imshow('test1', testImg)
    # cv2.waitKey(0)
    return dstImg

def fisheye_calib_normal_3(img, center_x, center_y, radius, size=0.7, zoom=1.5):
    height, width, _ = img.shape
    dst_size = round(radius * size * zoom)
    dstImg = np.zeros(((dst_size*2), (dst_size*2), 3), np.uint8)

    #求ct和r
    tmp_arr = (np.arange(dst_size) + 1) / zoom
    r = ((tmp_arr * tmp_arr + radius**2) / (2 * tmp_arr))
    ct = (tmp_arr - r)

    x1 = np.repeat(ct.reshape(1, -1), dst_size, axis=0)#(dst_size, dst_size)
    y2 = np.repeat(ct.reshape(-1, 1), dst_size, axis=1)#(dst_size, dst_size)
    r1 = np.repeat(r.reshape(1, -1), dst_size, axis=0)#(dst_size, dst_size)
    r2 = np.repeat(r.reshape(-1, 1), dst_size, axis=1)#(dst_size, dst_size)

    L_pow = x1 ** 2 + y2 ** 2#(dst_size, dst_size)
    K2 = x1 / y2#(dst_size, dst_size)
    r1_pow = r1 ** 2#(dst_size, dst_size)
    r2_pow = r2 ** 2#(dst_size, dst_size)
    AE = (r1_pow - r2_pow + L_pow) / (2 * L_pow)#(dst_size, dst_size)
    x0 = (1 - AE) * x1#(dst_size, dst_size)
    y0 = AE * y2#(dst_size, dst_size)
    CE = np.sqrt(r1_pow - (x0 - x1) ** 2 - y0 ** 2)#(dst_size, dst_size)

    x_ = (x0 + CE / np.sqrt(1 + K2 ** 2))#(dst_size, dst_size)
    y_ = (y0 + K2 * (x_ - x0))#(dst_size, dst_size)
    coef_x = x_ - x_.astype(np.int)#(dst_size, dst_size)
    coef_y = y_ - y_.astype(np.int)#(dst_size, dst_size)

    coef_x = np.repeat(coef_x, 3, axis=1).reshape(coef_x.shape[0], coef_x.shape[1], -1)
    coef_y = np.repeat(coef_y, 3, axis=1).reshape(coef_y.shape[0], coef_y.shape[1], -1)

    x00 = x_.astype(np.int) + int(center_x)
    y00 = y_.astype(np.int) + int(center_y)
    dstImg[dst_size : (dst_size * 2), dst_size : (dst_size * 2)] = (1 - coef_x) * (1 - coef_y) * img[y00, x00] + \
                                                                         coef_x * (1 - coef_y) * img[y00, x00 + 1] + \
                                                                         (1 - coef_x) * coef_y * img[y00 + 1, x00] + \
                                                                         coef_x * coef_y * img[y00 + 1, x00 + 1]

    x00 = (-x_).astype(np.int) + int(center_x)
    y00 = (y_).astype(np.int) + int(center_y)
    dstImg[dst_size:(dst_size * 2), (dst_size - 1)::-1] = (1 - coef_x) * (1 - coef_y) * img[y00, x00] + \
                                                  coef_x * (1 - coef_y) * img[y00, x00 - 1] + \
                                                  (1 - coef_x) * coef_y * img[y00 + 1, x00] + \
                                                  coef_x * coef_y * img[y00 + 1, x00 - 1]

    x00 = (x_).astype(np.int) + int(center_x)
    y00 = (-y_).astype(np.int) + int(center_y)
    dstImg[(dst_size - 1)::-1, dst_size:(dst_size * 2)] = (1 - coef_x) * (1 - coef_y) * img[y00, x00] + \
                                             coef_x * (1 - coef_y) * img[y00, x00 + 1] + \
                                             (1 - coef_x) * coef_y * img[y00 - 1, x00] + \
                                             coef_x * coef_y * img[y00 - 1, x00 + 1]

    x00 = (-x_).astype(np.int) + int(center_x)
    y00 = (-y_).astype(np.int) + int(center_y)
    dstImg[(dst_size - 1)::-1, (dst_size - 1)::-1] = (1 - coef_x) * (1 - coef_y) * img[y00, x00] + \
                                                 coef_x * (1 - coef_y) * img[y00, x00 - 1] + \
                                                 (1 - coef_x) * coef_y * img[y00 - 1, x00] + \
                                                 coef_x * coef_y * img[y00 - 1, x00 - 1]
    return dstImg

def fisheye_calib_normal_2(img, center_x, center_y, radius, size=0.7, zoom=1.5):
    height, width = img.shape[0:2]
    dst_size = int(radius * size * zoom)
    testImg = np.zeros((height, width), np.uint8)
    dstImg = np.zeros(((dst_size*2+1), (dst_size*2+1), 3), np.uint8)

    ct = np.zeros(dst_size)
    r = np.zeros(dst_size)
    for i in range(dst_size):
        j = (i+1) / zoom
        r[i] = (j*j + radius*radius) / (2*j)
        ct[i] = j - r[i]

    for i in range(dst_size):
        for j in range(dst_size):
            x1 = ct[j]#列
            r1 = r[j]#列
            y2 = ct[i]#行
            r2 = r[i]#行

            L_pow = x1*x1 + y2*y2
            K2 = x1/y2
            r1_pow = r1*r1
            r2_pow = r2*r2
            AE = (r1_pow - r2_pow + L_pow) / (2*L_pow)
            x0 = (1-AE)*x1
            y0 = AE*y2
            CE = math.sqrt(r1_pow - (x0-x1)*(x0-x1) - y0*y0)

            x_ = x0 + CE / math.sqrt(1 + K2*K2)
            y_ = y0 + K2*(x_ - x0)

            coef_x = x_ - int(x_)
            coef_y = y_ - int(y_)
            x00 = int(x_) + int(center_x)
            y00 = int(y_) + int(center_y)
            dstImg[dst_size + i][dst_size + j] = (1-coef_x)*(1-coef_y)*img[y00][x00] +\
                                                 coef_x * (1 - coef_y) * img[y00][x00+1] +\
                                                 (1-coef_x) * coef_y * img[y00+1][x00] +\
                                                 coef_x * coef_y * img[y00+1][x00+1]

            x00 = int(-x_) + int(center_x)
            y00 = int(y_) + int(center_y)
            dstImg[dst_size + i][dst_size - j - 1] = (1 - coef_x) * (1 - coef_y) * img[y00][x00] + \
                                                     coef_x * (1 - coef_y) * img[y00][x00 - 1] + \
                                                     (1 - coef_x) * coef_y * img[y00 + 1][x00] + \
                                                     coef_x * coef_y * img[y00 + 1][x00 - 1]

            x00 = int(x_) + int(center_x)
            y00 = int(-y_) + int(center_y)
            dstImg[dst_size - i - 1][dst_size + j] = (1 - coef_x) * (1 - coef_y) * img[y00][x00] + \
                                                     coef_x * (1 - coef_y) * img[y00][x00 + 1] + \
                                                     (1 - coef_x) * coef_y * img[y00 - 1][x00] + \
                                                     coef_x * coef_y * img[y00 - 1][x00 + 1]

            x00 = int(-x_) + int(center_x)
            y00 = int(-y_) + int(center_y)
            dstImg[dst_size - i - 1][dst_size - j - 1] = (1 - coef_x) * (1 - coef_y) * img[y00][x00] + \
                                                         coef_x * (1 - coef_y) * img[y00][x00 - 1] + \
                                                         (1 - coef_x) * coef_y * img[y00 - 1][x00] + \
                                                         coef_x * coef_y * img[y00 - 1][x00 - 1]
    return dstImg


images = glob.glob('/Users/startdt-algorithm/all_ciga_data/origin/*.jpg')
random.shuffle(images)
st = time.time()
for fname in images:
    img = cv2.imread(fname)

    centers, radius = GetMinCircle(img)

    calib_img = fisheye_calib_normal_3(img, centers[0], centers[1], radius)
    # cv2.imshow('test2', calib_img)
    # cv2.waitKey(0)

    base_filename = fname.split('.')[0]
    ffname = base_filename.split('/')[-1]
    print(ffname)
    # cv2.imwrite(base_filename + '_src.jpg', img)
    cv2.imwrite('/Users/startdt-algorithm/all_ciga_data/jiaozheng/' + ffname + '_calib.jpg', calib_img)

print('----spend time: ', time.time() - st)


