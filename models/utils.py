"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""
# default libraries
import os

# 3rd party libraries
import tensorflow as tf
import numpy as np

# local libraries

"""
Must set up an environment variable 'NUSCENES_DIR' in your OS with the directory of your NuScenes database
e.g. NUSCENES_DIR = C:/Data/NuScenes
"""
data_dir = os.environ.get('NUSCENES_DIR')


def load_fused_imgs_dataset():
    """

    Args:
        sample: shape (32, 1600, 900, 3)

    Returns:
        train_dataset:
        val_dataset:
    """

    fused_imgs_dir = os.path.join(data_dir, 'fused_imgs/')
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(fused_imgs_dir, labels=None,
                                                                        validation_split=0.2,
                                                                        subset="training",
                                                                        seed=11,
                                                                        image_size=(1600, 900),
                                                                        batch_size=1)
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(fused_imgs_dir, labels=None,
                                                                      validation_split=0.2,
                                                                      subset="validation",
                                                                      seed=11,
                                                                      image_size=(1600, 900),
                                                                      batch_size=1)


    bbox = []
    class_id = 0
    return train_dataset, val_dataset