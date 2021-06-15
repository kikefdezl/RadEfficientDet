"""

Written by Enrique Fernández-Laguilhoat Sánchez-Biezma

"""

# 3rd party libraries
import cv2
import numpy as np


def draw_overlay(image, points, depths, velocities):
    points = points.transpose()

    new_img = image
    for i, point in enumerate(points):
        point_x, point_y, point_z = point.astype(int)
        vel_x, vel_y = velocities[:, i]
        new_img = draw_circle(image, point_x, point_y, depths[i])
        new_img = draw_vector(image, point_x, point_y, depths[i], vel_x, vel_y)

    return new_img


def draw_circle(image, point_x, point_y, depth, radius: int = 4):

    bgr_color = get_depth_color(depth)

    new_img = cv2.circle(image, (point_x, point_y), radius, bgr_color, -1)

    return new_img


def draw_vector(image, point_x, point_y, depth, vel_x, vel_y, thickness: int=2):

    """
    Draws a vector showing the velocity of the point on the image
    Args:
        image: RGB image data
        point_x: integer x value of the pixel the origin of the vector is located at
        point_y: integer y value of the pixel de origin of the vector is located at
        depth: depth of the point (used to determine vector color)
        vel_x: speed of the point compensated by the car ego motion (X IS FRONT)
        vel_y: speed of the point compensated by the car ego motion (Y IS RIGHT)
        thickness: thickness of the vector

    Returns:
        new_img: modified image
    """

    bgr_color = get_depth_color(depth)

    vec_x_size = int(vel_y * 10)
    vec_y_size = int(vel_x * 10)

    pt1 = (point_x, point_y)
    pt2 = (point_x + vec_x_size, point_y + vec_y_size)
    new_img = cv2.line(image, pt1, pt2, bgr_color, thickness)

    return new_img

def get_depth_color(depth=0.0):
    """
    Returns BGR color values that vary depending on the depth of the point
    Args:
        depth: float value with the radar point depth

    Returns:
        bgr_color: a 3 value tuple, indicating the BGR color values (range [0, 255])
    """
    hue_value = min((depth / 255) * 179, 150)
    hsv_color = np.float32([[[hue_value, 255.0, 255.0]]]).astype(np.uint8)
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    bgr_color = np.reshape(bgr_color, (3,))
    bgr_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

    return bgr_color
