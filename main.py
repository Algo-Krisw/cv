# -*— coding: utf-8 -*_
# @Time    : 2021/6/17 10:37 下午
# @Author  : Algo
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import os

import cv2
import numpy as np
import math

# from libsvm.python.svmutil import *
# from libsvm.python.svm import *
import tqdm


def get_feature(path):
    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(gray_img.shape)
    ret, img = cv2.threshold(gray_img, 12, 240, cv2.THRESH_BINARY)
    # cv2.imshow("img", img)
    # cv2.waitKey(2)
    row, raw = img.shape
    print(img.shape)
    contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算第一条轮廓的各阶矩,字典形式
    m = cv2.moments(contours[0])
    # weighted center c_x & c_y
    c_x = int(m["m10"] / m["m00"])
    c_y = int(m["m01"] / m["m00"])
    img[c_y][c_x] = 0
    # poi数组存储点，k为角度
    print(contours[0])
    pt = [[0, 0] for i in range(270)]
    angle = []
    for i in range(len(contours[0])):
        if int(contours[0][i][0][0]) != c_x:
            angle.append(math.atan(
                (int(contours[0][i][0][1]) - c_y) / (int(contours[0][i][0][0]) - c_x)) * 180 / np.pi)
    i = 0
    while True:
        if -45 < angle[i] < 90:
            break
        else:
            pt[-int(round((angle[i]))) + 180][0] = int(contours[0][i][0][0])
            pt[-int(round((angle[i]))) + 180][1] = int(contours[0][i][0][1])
            i += 1
    while angle[i] > -45:
        i += 1
        pt[-int(round((angle[i]))) + 45][0] = int(contours[0][i][0][0])
        pt[-int(round((angle[i]))) + 45][1] = int(contours[0][i][0][1])
    i = -1
    while True:
        if -90 < angle[i] < 45:
            break
        else:
            pt[-int(round((angle[i]))) + 90][0] = int(contours[0][i][0][0])
            pt[-int(round((angle[i]))) + 90][1] = int(contours[0][i][0][1])
            i -= 1
    while angle[i] < -45:
        i -= 1
        pt[-int(round((angle[i]))) + 90][0] = int(contours[0][i][0][0])
        pt[-int(round((angle[i]))) + 90][1] = int(contours[0][i][0][1])
    for i in range(136):
        if pt[i][0] == 0:
            for j in range(i, 200):
                if pt[j][0] != 0:
                    pt[i] = pt[j]
                    break
    for i in range(269, 135, -1):
        if pt[i][0] == 0:
            for j in range(i, 0, -1):
                if pt[j][0] != 0:
                    pt[i] = pt[j]
                    break
    print(pt)
    character = [[] for i in range(10000)]
    for i in range(len(pt)):
        for j in range(1, 4):  # 三种不同的邻域
            temp_img = np.zeros(img.shape, dtype=img.dtype)
            # 将局部信息先存储起来，减少计算
            for k in range(int(pt[i][0]) - 8 * j, int(pt[i][0]) + 8 * j):
                for m in range(int(pt[i][1]) - 8 * j, int(pt[i][1]) + 8 * j):
                    if (k - int(pt[i][0])) ** 2 + (m - (int(pt[i][1]))) ** 2 < (8 * j) ** 2 \
                            and 0 <= k < raw and 0 <= m < row:
                        if img[m][k] == 255:
                            temp_img[k][m] = 255
            copy = temp_img.copy()  # 复制检测图像
            zero = np.zeros(temp_img.shape, np.uint8)  # 与检测图像相同，像素值为0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
            print(pt[i])
            zero[pt[i][1]][pt[i][0]] = 255  # 以圆心为起点
            # 对第一个点开始膨胀，再执行交集操作
            for p in range(200):
                dilation_B = cv2.dilate(zero, kernel, iterations=1)
                zero = cv2.bitwise_and(temp_img, dilation_B)
                # 取检测图像值为255的像素坐标，并将copy中对应坐标像素值变为0
            Xb, Yb = np.where(zero > 0)
            copy[Xb, Yb] = 0
            # 连通分量及其包含像素数量
            character[i].append(len(Xb))
            # 寻找主段
            left = 0
            right = 0
            # 向左查找边界，left表示左侧点的个数
            p = i
            q = p
            while (pt[p][0] - pt[q][0]) ** 2 + (pt[p][1] - pt[q][1]) ** 2 < (j * 8) ** 2:
                left += 1
                q -= 1
            # 向右查找边界，right表示右侧点的个数
            p = i
            q = p
            while 0 <= q < 270 and(pt[p][0] - pt[q][0]) ** 2 + (pt[p][1] - pt[q][1]) ** 2 < (j * 8) ** 2:
                right += 1
                q += 1
            # left和right的和即主段特征
            sum_length = right + left
            character[i].append(sum_length)
            # 计算中心距离特征
            dist = math.sqrt((int(pt[i][0]) - c_x) ** 2 + (int(pt[i][0]) - c_x) ** 2)
        character[i].append(dist)
    return character


path = "./ROI/1/6.bmp"
fea = get_feature(path)
fea = np.array([i for i in fea if i != []])
print(fea)
# pwd = os.getcwd()
# root = os.path.join(pwd, "ROI")
# for i in tqdm.tqdm(range(10)):
#     root_ges = os.path.join(root, str(i))
#     files = sorted(os.listdir(root_ges))
#     for file in files:
#         path = os.path.join(root_ges, file)
#         data = compute_character(path)
#         data = np.array([i for i in data if i != []])
#         np.savetxt("1.txt", data)
#         print(data)
