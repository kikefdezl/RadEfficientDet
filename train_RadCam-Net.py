"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""
#libraries
import os
import random

#local
from config import config
from nn_models.retinanet.keras_retinanet.preprocessing.csv_generator import CSVGenerator
from nn_models.retinanet.keras_retinanet import models, losses

#3rd pary libraries
import tensorflow as tf


def main():
    data_dir = config['data_dir']
    dataset_version = config['dataset_version']
    assert dataset_version == 'v1.0-mini' or dataset_version == 'v1.0-trainval', "The specified dataset version does " \
                                                                                 "not exist. Select 'mini' or " \
                                                                                 "'trainval'. "
    dataset_version_dir = os.path.join(data_dir, dataset_version)

    csv_train_data_file_path = os.path.join(dataset_version_dir, 'train_dataset.csv')
    csv_val_data_file_path = os.path.join(dataset_version_dir, 'val_dataset.csv')
    csv_class_file_path = os.path.join(dataset_version_dir, 'dataset_encoding.csv')
    train_dataset = CSVGenerator(csv_train_data_file_path, csv_class_file_path)
    val_dataset = CSVGenerator(csv_val_data_file_path, csv_class_file_path)

    model = models.backbone('resnet50').retinanet(num_classes=23)

    model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=tf.keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
    )

    model.summary()

    model.fit(x=train_dataset, validation_data=val_dataset, epochs=1)

    model.save_model('nn_models/saved/RadCamNet')

if __name__ == '__main__':
    main()