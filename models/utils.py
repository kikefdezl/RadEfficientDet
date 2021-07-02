"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""
# default libraries
import os

# 3rd party libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from nuscenes.nuscenes import NuScenes
import nuscenes.scripts.export_2d_annotations_as_json as e2daaj

# local libraries
from dataset_preprocessing.fusion import Fuser

"""
Must set up an environment variable 'NUSCENES_DIR' in your OS with the directory of your NuScenes database
e.g. NUSCENES_DIR = C:/Data/NuScenes
"""
data_dir = os.environ.get('NUSCENES_DIR')
fused_imgs_dir = os.path.join(data_dir, 'fused_imgs/')

dataset_version = 'mini'


def load_fused_imgs_dataset():
    """

    Args:
        sample:

    Returns:
        train_dataset:
        val_dataset:
    """

    if dataset_version == 'mini':
        nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    elif dataset_version == 'trainval':
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_dir, verbose=True)
    else:
        print("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
        exit(0)

    fuser = Fuser(nusc)
    list_of_sample_tokens = fuser.get_sample_tokens()

    for sample_token in list_of_sample_tokens:
        image, bbox, class_id = get_labels(nusc, sample_token)

    return train_dataset, val_dataset

def get_labels(nusc, sample_token):
    """

    Args:
        nusc: NuScenes object of the dataset
        sample_token: token of the sample we wish to extract the labels from

    Returns:
        image: fused_img, in array RGB format (1600, 900, 3)
        bbox: list of bounding boxes, with shape (num_objects, 4), where each box if of format (x, y, width, height).
        class_id: list of the class id number, with shape (num_objects,)
    """
    image_path = os.path.join(fused_imgs_dir, sample_token + '.png')
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.keras.preprocessing.image.img_to_array(image)

    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['CAM_FRONT']
    boxes = nusc.get_boxes(sample_data_token)
    nusc.render_annotation(boxes[0].token)

    bbox = []
    class_id = []
    return image, bbox, class_id