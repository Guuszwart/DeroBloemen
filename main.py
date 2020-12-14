# written for python 3.7

import cv2
import numpy as np


def nothing():
    pass


img = cv2.imread('Dero black.jpeg')
img = img[0:435, :]
copy = img.copy()
height, width = img.shape[:2]

cv2.namedWindow('trackbars')

cv2.createTrackbar('L-H', 'trackbars', 0, 179, nothing)
cv2.createTrackbar('L-S', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('L-V', 'trackbars', 149, 255, nothing)
cv2.createTrackbar('U-H', 'trackbars', 179, 179, nothing)
cv2.createTrackbar('U-S', 'trackbars', 255, 255, nothing)
cv2.createTrackbar('U-V', 'trackbars', 255, 255, nothing)
upper_buffer = np.array([179, 255, 255])
lower_buffer = np.array([0, 0, 0])

while True:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Trackbars for the HSV upper and lower limits
    l_h = cv2.getTrackbarPos('L-H', 'trackbars')
    l_s = cv2.getTrackbarPos('L-S', 'trackbars')
    l_v = cv2.getTrackbarPos('L-V', 'trackbars')
    u_h = cv2.getTrackbarPos('U-H', 'trackbars')
    u_s = cv2.getTrackbarPos('U-S', 'trackbars')
    u_v = cv2.getTrackbarPos('U-V', 'trackbars')

    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    conditions = [
        upper_range[0] != upper_buffer[0],
        upper_range[1] != upper_buffer[1],
        upper_range[2] != upper_buffer[2],
        lower_range[0] != lower_buffer[0],
        lower_range[1] != lower_buffer[1],
        lower_range[2] != lower_buffer[2]
    ]
# print the new HSV values is the sliders are changed
    if any(conditions):
        print(lower_range, upper_range)
        upper_buffer, lower_buffer = upper_range, lower_range
    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(img, img, mask=mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    erode = cv2.erode(res, kernel, iterations=1)
    erode = cv2.erode(erode, kernel, iterations=1)
    erode = cv2.erode(erode, kernel, iterations=1)
    erode = cv2.erode(erode, kernel, iterations=1)
    erode = cv2.erode(erode, kernel, iterations=1)
    erode2 = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(erode2, 10, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arr = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        arr.append(area)

    pos = (arr.index(max(arr)))
    cnt = contours[pos]
    M = cv2.moments(cnt)

# Calculate the centroid of the contour
    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

# Determine the highest point of the contour
    top = tuple(cnt[cnt[:, :, 1].argmin()][0])

# draw shapes into the existing image
    cv2.drawContours(copy, contours, pos, (255, 19, 25), 3)
    cv2.circle(copy, (cx, cy), 5, (0, 0, 255), thickness=-1, lineType=8, shift=0)
    cv2.line(copy, (0, cy), (width, cy), (0, 0, 255), 2)
    cv2.circle(copy, top, 5, (0, 255, 0), thickness=-1, lineType=8, shift=0)
    cv2.line(copy, (0, top[1]), (width, top[1]), (0, 255, 0), 2)

# show images
    cv2.imshow('cutout', thresh)
    cv2.imshow('original', img)
    cv2.imshow('Flowers', copy)

    if cv2.waitKey(1) == 27:
        break
