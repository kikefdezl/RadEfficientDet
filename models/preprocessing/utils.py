"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""
# default libraries
import os
import json

# 3rd party libraries
import tensorflow as tf
import numpy as np
from nuscenes.nuscenes import NuScenes
import cv2
from tqdm import tqdm

# local libraries
from dataset_preprocessing.fusion import Fuser
from dataset_preprocessing.image_graphics import draw_bbox

"""
Must set up an environment variable 'NUSCENES_DIR' in your OS with the directory of your NuScenes database
e.g. NUSCENES_DIR = C:/Data/NuScenes
"""
data_dir = os.environ.get('NUSCENES_DIR')
fused_imgs_dir = os.path.join(data_dir, 'fused_imgs/')

dataset_version = 'mini'  # set as 'mini' for troubleshooting (reduces load times)


def load_fused_imgs_dataset():
    """
    Returns training and validation datasets from the fused radar/camera images, including bounding boxes and class ids.

    Args:

    Returns:
        train_dataset:
        val_dataset:
    """

    if dataset_version == 'mini':
        nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    elif dataset_version == 'trainval':
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_dir, verbose=True)
    else:
        raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
    fuser = Fuser(nusc)
    list_of_sample_tokens = fuser.get_sample_tokens()

    print("Loading the dataset annotations:")
    formatted_dataset = []
    for sample_token in tqdm(list_of_sample_tokens):
        image, bboxes, class_ids = get_labels(nusc, sample_token)
        formatted_dataset.append([image, bboxes, class_ids])

    return formatted_dataset


def get_labels(nusc, sample_token):
    """
    Gets the list of bounding boxes and class ids for a specific sample

    Args:
        nusc: NuScenes object of the dataset
        sample_token: token of the sample we wish to extract the labels from

    Returns:
        image: fused_img, in array RGB format (1600, 900, 3)
        bbox: list of bounding boxes, with shape (num_objects, 4), where each box if of format (x, y, width, height).
        class_id: list of the class id number, with shape (num_objects,)
    """

    # get image
    image_path = os.path.join(fused_imgs_dir, sample_token + '.png')
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.keras.preprocessing.image.img_to_array(image)

    # get bounding boxes
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['CAM_FRONT']

    if dataset_version == 'mini':
        anns_dir = os.path.join(data_dir, 'v1.0-mini', 'image_annotations.json')
    elif dataset_version == 'trainval':
        anns_dir = os.path.join(data_dir, 'v1.0-trainval', 'image_annotations.json')
    else:
        raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")

    with open(anns_dir, 'r') as anns_file:
        data = json.load(anns_file)

    bboxes = []
    class_ids = []
    for annotation in data:
        if annotation['sample_data_token'] == sample_data_token:
            bounding_box = annotation['bbox_corners']
            bbox_x = bounding_box[0]
            bbox_y = bounding_box[1]
            bbox_width = bounding_box[2] - bounding_box[0]
            bbox_height = bounding_box[3] - bounding_box[1]

            bboxes.append([bbox_x, bbox_y, bbox_width, bbox_height])
            encoded_class = encode_class_id(annotation['category_name'])
            class_ids.append(encoded_class)

    """
    Uncomment the next section for rendering the 2D bboxes and classes (for testing purposes)
    """
    # new_img = cv2.imread(image_path)
    # for i, bbox in enumerate(bboxes):
    #     bbox_x = bbox[0]
    #     bbox_y = bbox[1]
    #     bbox_width = bbox[2]
    #     bbox_height = bbox[3]
    #     new_img = draw_bbox(new_img, bbox_x, bbox_y, bbox_width, bbox_height, class_id=class_ids[i])
    # cv2.imshow('window', new_img)
    # cv2.waitKey()

    return image, bboxes, class_ids


def encode_class_id(class_id):
    encoder = {
        'animal': 1,
        'human.pedestrian.adult': 2,
        'human.pedestrian.child': 3,
        'human.pedestrian.construction_worker': 4,
        'human.pedestrian.personal_mobility': 5,
        'human.pedestrian.police_officer': 6,
        'human.pedestrian.stroller': 7,
        'human.pedestrian.wheelchair': 8,
        'movable_object.barrier': 9,
        'movable_object.debris': 10,
        'movable_object.pushable_pullable': 11,
        'movable_object.trafficcone': 12,
        'static_object.bicycle_rack': 13,
        'vehicle.bicycle': 14,
        'vehicle.bus.bendy': 15,
        'vehicle.bus.rigid': 16,
        'vehicle.car': 17,
        'vehicle.construction': 18,
        'vehicle.emergency.ambulance': 19,
        'vehicle.emergency.police': 20,
        'vehicle.motorcycle': 21,
        'vehicle.trailer': 22,
        'vehicle.truck': 23
    }

    return encoder[class_id]
