# 3rd party libraries
import cv2
import numpy as np


def draw_overlay(image, points, coloring, radar_image):

    velocities = [radar_image[8], radar_image[9]]
    points = points.transpose()

    new_img = image
    for i, point in enumerate(points):
        new_img = draw_circles(image, point, coloring[i])

    return new_img


def draw_circles(image, point, coloring):
    point_x, point_y, point_z = point  # point_z not used
    point_x = int(point_x)
    point_y = int(point_y)

    hue_value = min((coloring / 255) * 179, 150)
    hsv_color = np.float32([[[hue_value, 255.0, 255.0]]]).astype(np.uint8)
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    bgr_color = np.reshape(bgr_color, (3,))
    bgr_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
    new_img = cv2.circle(image, (point_x, point_y), 3, bgr_color, -1)

    return new_img


def draw_vectors(image, point, vel_x, vel_y):
    return 0
