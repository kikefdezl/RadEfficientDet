"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""
#libraries
import os

#local
from config import config
from models.retinanet.keras_retinanet.preprocessing.csv_generator import CSVGenerator

#3rd pary libraries
import tensorflow as tf


def main():
    data_dir = config['data_dir']
    dataset_version = config['dataset_version']
    assert dataset_version == 'v1.0-mini' or dataset_version == 'v1.0-trainval', "The specified dataset version does " \
                                                                                 "not exist. Select 'mini' or " \
                                                                                 "'trainval'. "
    dataset_version_dir = os.path.join(data_dir, dataset_version)

    csv_data_file_path = os.path.join(dataset_version_dir, 'dataset.csv')
    csv_class_file_path = os.path.join(dataset_version_dir, 'dataset_encoding.csv')
    csv_generator = CSVGenerator(csv_data_file_path, csv_class_file_path)

    print(csv_generator.size())

if __name__ == '__main__':
    main()