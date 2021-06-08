
# 3rd party libraries
import cv2
import numpy as np


def draw_circles(image, points):
    new_img = image
    points = points.transpose()
    for point in points:
        point_x, point_y, point_z = point
        point_x = int(point_x)
        point_y = int(point_y)
        new_img = cv2.circle(new_img, (point_x, point_y), 5, (0,255,0), 4)
    cv2.imshow('windowname', new_img)
    cv2.waitKey(50)