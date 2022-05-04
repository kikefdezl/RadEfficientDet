"""
Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

This file is used to render the annotations from a CSV file on images and view them with OpenCV.
"""

import yaml
import os
import nuscenes
import json
import cv2
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
import numpy as np


def render_cv2(box,
               im: np.ndarray,
               view: np.ndarray = np.eye(3),
               normalize: bool = True,
               colors=((0, 255, 255), (0, 255, 255), (0, 255, 255)),
               linewidth: int = 2) -> None:
    """
    Renders box using OpenCV2.
    :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
    :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
    :param linewidth: Linewidth for plot.
    """
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(im,
                     (int(prev[0]), int(prev[1])),
                     (int(corner[0]), int(corner[1])),
                     color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(im,
                 (int(corners.T[i][0]), int(corners.T[i][1])),
                 (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                 colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0][::-1])
    draw_rect(corners.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(im,
             (int(center_bottom[0]), int(center_bottom[1])),
             (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
             colors[0][::-1], linewidth)

    return im


def render_3d_bboxes(nusc, image, sample_data_token: str, box_vis_level: BoxVisibility = BoxVisibility.ANY) -> None:

    # Load boxes and image.
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token,
                                                                   box_vis_level=box_vis_level)

    for box in boxes:
        annotation = nusc.get('sample_annotation', box.token)
        if annotation['visibility_token'] == '4':
            image = render_cv2(box, image, view=camera_intrinsic)

    return image

def get_list_of_images(nusc, camera_side):
    list_of_samples = nusc.sample
    list_of_camera_data_tokens = [sample['data'][camera_side] for sample in list_of_samples]
    list_of_camera_filenames = [nusc.get('sample_data', token)['filename'] for token in list_of_camera_data_tokens]

    return list_of_camera_filenames, list_of_camera_data_tokens

def render_2d_bboxes(image, annotations):
    for annotation in annotations:
        x1, y1, x2, y2 = [int(ann) for ann in annotation['bbox_corners']]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)

    return image


def main():
    with open('../config.yaml') as yf:
        config = yaml.safe_load(yf)

    nuscenes_dir = config['nuscenes_dir']
    dataset_save_dir = config['dataset_save_dir']

    assert os.path.exists(os.path.join(config['dataset_save_dir'], 'image_annotations.json')), "Error, make sure you" \
                                                                                               "generated the 'image_annotations.json' file first, by running export_2d_annotations_as_json.py"

    nusc = nuscenes.NuScenes(version=config['dataset_version'], dataroot=nuscenes_dir)

    with open(os.path.join(dataset_save_dir, 'image_annotations.json')) as jf:
        data_2d = json.load(jf)

    camera_side = config['COMPARE_3D_AND_2D_BBOXES']['camera_side']
    list_of_images, list_of_tokens = get_list_of_images(nusc, camera_side)

    for imagepath, token in zip(list_of_images, list_of_tokens):
        image_2d = cv2.imread(os.path.join(nuscenes_dir, imagepath))
        image_3d = cv2.imread(os.path.join(nuscenes_dir, imagepath))
        annotations = [ann for ann in data_2d if ann['filename'] == imagepath]

        image_2d = render_2d_bboxes(image_2d, annotations)
        image_3d = render_3d_bboxes(nusc, image_3d, token)

        final_image = cv2.hconcat([image_3d, image_2d])

        # resize the image to fit the screen
        ratio = final_image.shape[0] / final_image.shape[1]
        final_image = cv2.resize(final_image, (1920, int(1920*ratio)))

        cv2.imshow('3D_2D_Comparison', final_image)

        if cv2.waitKey(0) == 27:
            break



if __name__ == "__main__":
    main()
