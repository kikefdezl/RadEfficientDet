"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""

# 3rd party libraries
import cv2
import numpy as np
from math import atan, cos, sin, pi


def draw_overlay(image, points, depths, velocities, max_distance=1e09):
    points = points.transpose()

    new_img = image
    for i, point in enumerate(points):
        if depths[i] < max_distance:
            point_x, point_y, point_z = point.astype(int)
            vel_x, vel_y = velocities[:, i]
            new_img = draw_depth_circle(image, point_x, point_y, depths[i])
            new_img = draw_depth_vector(new_img, point_x, point_y, depths[i], vel_x, vel_y)

    return new_img

def draw_overlay_v2(image, points, depths, velocities, max_distance=1e09):
    points = points.transpose()
    velocities = velocities.transpose()

    overlay = image.copy()
    for point, depth, velocity in zip(points, depths, velocities):
        if depth < max_distance:
            point_x, point_y, point_z = point.astype(int)
            vel_x, vel_y = velocity
            radius = int(250/depth)
            overlay = draw_depth_circle(overlay, point_x, point_y, depth, radius)
            overlay = draw_depth_vector(overlay, point_x, point_y, depth, vel_x, vel_y)

    new_img = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    return new_img

def draw_radar_maps(image, points, depths, velocities, n_layers=5, radar_range=(0, 250)):
    points = points.transpose()
    velocities = velocities.transpose()
    # divide the 0-250m range exponentially
    n_divisions = sum([2**i for i in range(n_layers-1)])
    meters_per_division = radar_range[1]/n_divisions
    values = [(2**i) * meters_per_division for i in range(n_layers-1)]
    values.insert(0, 0.0)
    values.append(99999999.9)

    image_h = image.shape[0]
    image_w = image.shape[1]
    maps = []
    for idx, val in enumerate(values[1:]):
        applicables = [(depth < val) and (depth > values[idx]) for depth in depths]
        applicable_depths = depths[applicables]
        applicable_points = points[applicables]
        applicable_velocities = velocities[applicables]

        map = np.zeros((image_h, image_w, 3), np.uint8)
        for point, depth, velocity in zip(applicable_points, applicable_depths, applicable_velocities):
            point_x, point_y, point_z = point.astype(int)
            vel_x, vel_y = velocity
            radius = 25
            map = draw_circle(map, point_x, point_y, radius)
            map = draw_vector(map, point_x, point_y, vel_x, vel_y)

        maps.append(map)

    return maps

    # return maps

def draw_radar_maps_v2(image, points, depths, velocities, n_layers=5, radar_range=(0, 250), checkerboard_path=None):
    points = points.transpose()
    velocities = velocities.transpose()
    # divide the 0-250m range exponentially
    n_divisions = sum([2**i for i in range(n_layers-1)])
    meters_per_division = radar_range[1]/n_divisions
    values = [(2**i) * meters_per_division for i in range(n_layers-1)]
    values.insert(0, 0.0)
    values.append(99999999.9)
    if checkerboard_path:
        checkerboard_img = cv2.imread(checkerboard_path)

    image_h = image.shape[0]
    image_w = image.shape[1]
    maps = []
    for idx, val in enumerate(values[1:]):
        applicables = [(depth < val) and (depth > values[idx]) for depth in depths]
        applicable_depths = depths[applicables]
        applicable_points = points[applicables]
        applicable_velocities = velocities[applicables]

        map = np.zeros((image_h, image_w, 3), np.uint8)
        for point, depth, velocity in zip(applicable_points, applicable_depths, applicable_velocities):
            point_x, point_y, point_z = point.astype(int)
            vel_x, vel_y = velocity
            radius = int(256 // depth)
            map = draw_circle(map, point_x, point_y, radius)
            map = draw_vector(map, point_x, point_y, vel_x, vel_y)
        if checkerboard_path:
            map = cv2.bitwise_and(checkerboard_img, map)

        maps.append(map)

    return maps

def draw_radar_maps_v3(image, points, depths, velocities, n_layers=5, radar_range=(0, 250), checkerboard_path=None):

    """
    Function that creates 5 radar maps for different distances. Drawn as circles, where the size is calculated to be 1
    meter in diameter on the object its detecting.
    NuScenes camera has a 5.5mm focal length, 7.2mm horizontal sensor size, and a 1600px horizontal resolution. The
    funtion obtain the number of pixels for a 1 meter object on the image is:

    n_pix = (0.0055 x 1600)/(distance x 0.0072)

    :param image: TODO
    :param points: TODO
    :param depths: TODO
    :param velocities: TODO
    :param n_layers: TODO
    :param radar_range: TODO
    :param checkerboard_path: TODO
    :return: TODO
    """
    points = points.transpose()
    velocities = velocities.transpose()
    # divide the 0-250m range exponentially
    n_divisions = sum([2 ** i for i in range(n_layers - 1)])
    meters_per_division = radar_range[1] / n_divisions
    values = [(2 ** i) * meters_per_division for i in range(n_layers - 1)]
    values.insert(0, 0.0)
    values.append(99999999.9)
    if checkerboard_path:
        checkerboard_img = cv2.imread(checkerboard_path)

    image_h = image.shape[0]
    image_w = image.shape[1]
    maps = []
    for idx, val in enumerate(values[1:]):
        applicables = [(depth < val) and (depth > values[idx]) for depth in depths]
        applicable_depths = depths[applicables]
        applicable_points = points[applicables]
        applicable_velocities = velocities[applicables]

        map = np.zeros((image_h, image_w, 3), np.uint8)
        for point, depth, velocity in zip(applicable_points, applicable_depths, applicable_velocities):
            point_x, point_y, point_z = point.astype(int)
            vel_x, vel_y = velocity
            radius = int((0.0055 * 1600)/(2 * 0.0072 * depth))
            point_y -= radius
            map = draw_circle(map, point_x, point_y, radius)
            map = draw_vector(map, point_x, point_y, vel_x, vel_y)
        if checkerboard_path:
            map = cv2.bitwise_and(checkerboard_img, map)
        image = cv2.bitwise_or(map, image)

        maps.append(map)

    return maps


def draw_radar_maps_test(image, points, depths, velocities, n_layers=5, radar_range=(0, 250), checkerboard_path=None):
    """
    This function is for testing purposes

    n_pix = (0.0055 x 1600)/(distance x 0.0072)
    """
    points = points.transpose()
    velocities = velocities.transpose()
    # divide the 0-250m range exponentially
    n_divisions = sum([2 ** i for i in range(n_layers - 1)])
    meters_per_division = radar_range[1] / n_divisions
    values = [(2 ** i) * meters_per_division for i in range(n_layers - 1)]
    values.insert(0, 0.0)
    values.append(99999999.9)
    if checkerboard_path:
        checkerboard_img = cv2.imread(checkerboard_path)

    image_h = image.shape[0]
    image_w = image.shape[1]
    maps = []
    for idx, val in enumerate(values[1:]):
        applicables = [(depth < val) and (depth > values[idx]) for depth in depths]
        applicable_depths = depths[applicables]
        applicable_points = points[applicables]
        applicable_velocities = velocities[applicables]

        map = np.zeros((image_h, image_w, 3), np.uint8)
        for point, depth, velocity in zip(applicable_points, applicable_depths, applicable_velocities):
            point_x, point_y, point_z = point.astype(int)
            vel_x, vel_y = velocity
            radius = int((0.0055 * 1600)/(2 * 0.0072 * depth))
            point_y -= radius
            map = draw_circle(map, point_x, point_y, radius, color=(255, 255, 255))
            # map = draw_white_vector(map, point_x, point_y, vel_x, vel_y)
        if checkerboard_path:
            map = cv2.bitwise_and(checkerboard_img, map)
        image = cv2.bitwise_or(map, image)

        maps.append(map)

    return maps


def draw_depth_circle(image, point_x, point_y, depth, radius: int = 4):
    """
    Draws a circle on an image, and returns the new image
    Args:
        image: image object
        point_x: x coordinate of the center of the circle
        point_y: y coordinate of the center of the circle
        depth: distance from the camera of the point.
        radius: radius of the circle

    Returns:
        new_img: modified image
    """
    bgr_color = get_depth_color(depth)

    new_img = cv2.circle(image, (point_x, point_y), radius, bgr_color, -1)

    return new_img

def draw_circle(image, point_x, point_y, radius: int = 4, color=(255, 255, 255)):
    """
    Draws a circle on an image, and returns the new image
    Args:
        image: image object
        point_x: x coordinate of the center of the circle
        point_y: y coordinate of the center of the circle
        depth: distance from the camera of the point.
        radius: radius of the circle

    Returns:
        new_img: modified image
    """
    new_img = cv2.circle(image, (point_x, point_y), radius, color, -1)

    return new_img


def draw_depth_vector(image, point_x, point_y, depth, vel_x, vel_y, thickness: int = 2):
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
        new_img: modified image with superposed vector
    """

    bgr_color = get_depth_color(depth)

    vec_x_size = -int(vel_y * 10)
    vec_y_size = -int(vel_x * 10)

    pt1 = (point_x, point_y)
    pt2 = (point_x + vec_x_size, point_y + vec_y_size)
    new_img = cv2.line(image, pt1, pt2, bgr_color, thickness)

    vec_size = np.sqrt(vec_x_size ** 2 + vec_y_size ** 2)

    vector_angle = atan(vel_x / vel_y)
    if vel_y >= 0:
        vector_angle += pi

    arrow_end_1 = (pt2[0] - (cos(vector_angle + 0.35) * 10), pt2[1] - (sin(vector_angle + 0.35) * 10))
    arrow_end_2 = (pt2[0] - (cos(vector_angle - 0.35) * 10), pt2[1] - (sin(vector_angle - 0.35) * 10))

    arrow_end_1 = (int(arrow_end_1[0]), int(arrow_end_1[1]))
    arrow_end_2 = (int(arrow_end_2[0]), int(arrow_end_2[1]))
    if vec_size > 5:
        new_img = cv2.line(new_img, pt2, arrow_end_1, bgr_color, thickness)
        new_img = cv2.line(new_img, pt2, arrow_end_2, bgr_color, thickness)

    return new_img

def draw_vector(image, point_x, point_y, vel_x, vel_y, color=(255, 255, 255), thickness: int = 2):
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
        new_img: modified image with superposed vector
    """

    vec_x_size = -int(vel_y * 10)
    vec_y_size = -int(vel_x * 10)

    pt1 = (point_x, point_y)
    pt2 = (point_x + vec_x_size, point_y + vec_y_size)
    new_img = cv2.line(image, pt1, pt2, color, thickness)

    vec_size = np.sqrt(vec_x_size ** 2 + vec_y_size ** 2)

    vector_angle = atan(vel_x / vel_y)
    if vel_y >= 0:
        vector_angle += pi

    arrow_end_1 = (pt2[0] - (cos(vector_angle + 0.35) * 10), pt2[1] - (sin(vector_angle + 0.35) * 10))
    arrow_end_2 = (pt2[0] - (cos(vector_angle - 0.35) * 10), pt2[1] - (sin(vector_angle - 0.35) * 10))

    arrow_end_1 = (int(arrow_end_1[0]), int(arrow_end_1[1]))
    arrow_end_2 = (int(arrow_end_2[0]), int(arrow_end_2[1]))
    if vec_size > 5:
        new_img = cv2.line(new_img, pt2, arrow_end_1, (255, 255, 255), thickness)
        new_img = cv2.line(new_img, pt2, arrow_end_2, (255, 255, 255), thickness)

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


def draw_bbox(image, x: int, y: int, width: int, height: int, class_id: str = ''):
    """
    draws a 2D bounding box on an image, with the class category
    Args:
        image: image object (cv2.imread)
        x: x pixel of the top left corner
        y: y pixel of the top left corner
        width: horizontal pixels of the bounding box
        height: vertical pixels of the bounding box
        class_id: class id string to display over the bbox

    Returns:
        new_img: modified image
    """
    pt1 = (int(x), int(y))
    pt2 = (int(x + width), int(y + height))
    new_img = cv2.rectangle(image, pt1, pt2, color=(0, 255, 0), thickness=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_img = cv2.putText(new_img, str(class_id), pt1, font, 0.5, (255, 255, 255), 1)

    return new_img
