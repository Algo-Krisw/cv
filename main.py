import cv2
import numpy as np
import math
# from libsvm.python.svmutil import *
# from libsvm.python.svm import *


def compute_character(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", img)
    cv2.waitKey(2)
    row, raw = img.shape
    print(img.shape)
    contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算第一条轮廓的各阶矩,字典形式
    m = cv2.moments(contours[0])
    # 计算形状的中心
    center_x = int(m["m10"] / m["m00"])
    center_y = int(m["m01"] / m["m00"])
    img[center_y][center_x] = 0
    # character数组用于存储特征信息
    points = [[0, 0] for i in range(270)]
    angle = []
    for i in range(len(contours[0])):
        if int(contours[0][i][0][0]) != center_x:
            angle.append(math.atan(
                (int(contours[0][i][0][1]) - center_y) / (int(contours[0][i][0][0]) - center_x)) * 180 / 3.14159)
    i = 0
    while True:
        if -45 < angle[i] < 90:
            break
        else:
            points[-int(round((angle[i]))) + 180][0] = int(contours[0][i][0][0])
            points[-int(round((angle[i]))) + 180][1] = int(contours[0][i][0][1])
            i += 1
    while angle[i] > -45:
        i += 1
        points[-int(round((angle[i]))) + 45][0] = int(contours[0][i][0][0])
        points[-int(round((angle[i]))) + 45][1] = int(contours[0][i][0][1])
    i = -1
    while True:
        if -90 < angle[i] < 45:
            break
        else:
            points[-int(round((angle[i]))) + 90][0] = int(contours[0][i][0][0])
            points[-int(round((angle[i]))) + 90][1] = int(contours[0][i][0][1])
            i -= 1
    while angle[i] < -45:
        i -= 1
        points[-int(round((angle[i]))) + 90][0] = int(contours[0][i][0][0])
        points[-int(round((angle[i]))) + 90][1] = int(contours[0][i][0][1])
    for i in range(136):
        if points[i][0] == 0:
            for j in range(i, 200):
                if points[j][0] != 0:
                    points[i] = points[j]
                    break
    for i in range(269, 135, -1):
        if points[i][0] == 0:
            for j in range(i, 0, -1):
                if points[j][0] != 0:
                    points[i] = points[j]
                    break
    character = [[] for i in range(10000)]
    for i in range(len(points)):
        for j in range(1, 4):  # 三种不同的邻域
            temp_img = np.zeros(img.shape, dtype=img.dtype)
            # 将局部信息先存储起来，减少计算
            for k in range(int(points[i][0]) - 20 * j, int(points[i][0]) + 20 * j):
                for m in range(int(points[i][1]) - 20 * j, int(points[i][1]) + 20 * j):
                    if k > 0 and m > 0 and (k - int(points[i][0])) ** 2 + (m - (int(points[i][1]))) ** 2 < (
                            20 * j) ** 2 and m < row and k < raw:
                        if img[m][k] == 255 and m < raw and k < row:
                            temp_img[k][m] = 255
            copy = temp_img.copy()  # 复制检测图像
            zero = np.zeros(temp_img.shape, np.uint8)  # 与检测图像相同，像素值为0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
            # 腐蚀算法和二值图像的交集算法
            count = []
            bre = 0
            zero[points[i][0]][points[i][1]] = 255  # 以圆心为起点
            # 对第一个点开始膨胀，再执行交集操作
            for p in range(200):
                dilation_b = cv2.dilate(zero, kernel, iterations=1)
                zero = cv2.bitwise_and(temp_img, dilation_b)
                # 取检测图像值为255的像素坐标，并将copy中对应坐标像素值变为0
            x_b, y_b = np.where(zero > 0)
            copy[x_b, y_b] = 0
            # 连通分量及其包含像素数量
            character[i].append(len(x_b))
            # 寻找主段
            left = 0
            right = 0
            # 向左查找边界，left表示左侧点的个数
            p = i
            q = p
            while (points[p][0] - points[q][0]) ** 2 + (points[p][1] - points[q][1]) ** 2 < (j * 20) ** 2 and p < 270 and q < 270:
                print(p, q, j)
                left += 1
                q -= 1
            # 向右查找边界，right表示右侧点的个数
            p = i
            q = p
            while (points[p][0] - points[q][0]) ** 2 + (points[p][1] - points[q][1]) ** 2 < (j * 20) ** 2 and p < 270 and q < 270:
                right += 1
                q += 1
            # left和right的和即主段特征
            sum_length = right + left
            character[i].append(sum_length)
            # 计算中心距离特征
            dist = math.sqrt((int(points[i][0]) - center_x) ** 2 + (int(points[i][0]) - center_x) ** 2)
        character[i].append(dist)
    return character


path = "./ROI/0/10.bmp"
data = compute_character(path)
print(data)
