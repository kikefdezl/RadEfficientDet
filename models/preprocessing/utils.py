"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""
# default libraries
import os
import argparse

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

dataset_version = 'mini'  # set as 'mini' for troubleshooting (reduces load times)


def load_fused_imgs_dataset():
    """
    Returns training and validation datasets from the fused radar/camera images, including bounding boxes and class ids.

    Args:

    Returns:
        train_dataset:
        val_dataset:
    """

    # generate a json file with 2D bounding boxes
    parser = argparse.ArgumentParser(description='Export 2D annotations from reprojections to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default=data_dir, help="Path where nuScenes is saved.")
    parser.add_argument('--filename', type=str, default='fused_imgs_annotations.json', help='Output filename.')
    parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
                        help='Visibility bins, the higher the number the higher the visibility.', nargs='+')
    parser.add_argument('--image_limit', type=int, default=-1, help='Number of images to process or -1 to process all.')
    if dataset_version == 'mini':
        parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version.')
    elif dataset_version == 'trainval':
        parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version.')
    else:
        raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
    args = parser.parse_args()

    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    e2daaj.main(args)

    fuser = Fuser(nusc)
    list_of_sample_tokens = fuser.get_sample_tokens()

    for sample_token in list_of_sample_tokens:
        image, bbox, class_id = get_labels(nusc, sample_token)

    return train_dataset, val_dataset


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
