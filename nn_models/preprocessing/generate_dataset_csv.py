"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""
# default libraries
import os
import json
import csv
from random import shuffle
from time import time

# 3rd party libraries
from nuscenes.nuscenes import NuScenes
import cv2
from tqdm import tqdm

# local libraries
from dataset_preprocessing.image_graphics import draw_bbox
from config import config


def generate_dataset_csv(size_threshold=3, shuffle_data=True, validation_split: float = 0.2, overwrite=False):
    """
    Generates two csv files (training and validation) containing one annotation per row. As per RetinaNet instructions,
    the format of each row is:

    > path/to/image.jpg,x1,y1,x2,y2,class_name
    Args:
        overwrite       : set as True to overwrite files if they are already on disk. If not, the program will exit upon
                        finding the files
        validation_split: float value to determine the percentage of data to save to the validation CSV file
        shuffle_data    : set as true to shuffle all the parameters before creating the CSV file
        size_threshold  : bounding box height or width (in pixels), below which an annotation is deleted. Used for
                        filtering excessively small bounding boxes.

    """

    dataset_version = config['dataset_version']
    if dataset_version != 'v1.0-mini' and dataset_version != 'v1.0-trainval':
        raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
    data_dir = config['data_dir']
    dataset_version_dir = os.path.join(data_dir, dataset_version)
    print("Generating the dataset CSV for dataset version %s:" % dataset_version)

    # check if the CSV files are already there
    csv_train_file_dir = os.path.join(dataset_version_dir, 'train_dataset.csv')
    csv_val_file_dir = os.path.join(dataset_version_dir, 'val_dataset.csv')
    if not overwrite:
        if os.path.exists(csv_train_file_dir) and os.path.exists(csv_val_file_dir):
            print("======")
            print("Dataset CSV files already exist at:")
            print(csv_train_file_dir)
            print(csv_val_file_dir)
            print("======")
            return 0

    # check that the validation split is between 0 and 1
    assert 0.0 <= validation_split <= 1.0, "Incorrect validation_split value. It must be a float value between 0 & 1."

    nusc = NuScenes(version=dataset_version, dataroot=data_dir, verbose=True)

    anns_dir = os.path.join(dataset_version_dir, 'image_annotations.json')
    if not os.path.exists(anns_dir):
        raise Exception(
            "No annotation data. Must generate 2D annotation JSON file first. Run 'nn_models/preprocessing/export"
            "_2d_annotations_as_json.py'.")

    with open(anns_dir, 'r') as _2d_anns_file:
        print("Opening 2D annotations JSON file...")
        t_start = time()
        _2d_anns_data = json.load(_2d_anns_file)
        elapsed = round(time() - t_start, 2)
        print(f"Done opening 2D annotations file in {elapsed} seconds.")
        print("======")

        # remove annotations with width or height below the threshold
        print("Filtering 2D annotations...")
        index = 0
        removed = 0
        while index < len(_2d_anns_data):
            bbox_corners = _2d_anns_data[index]['bbox_corners']
            if ((bbox_corners[2] - bbox_corners[0]) < size_threshold or
                    (bbox_corners[3] - bbox_corners[1]) < size_threshold):
                _2d_anns_data.pop(index)
                removed += 1
            else:
                index += 1
        print(f"Removed {removed} annotations with width or height below {size_threshold} pixels.")
        print("======")

        # shuffle the data if desired, and split into 80% train, 20% validation
        if shuffle_data:
            shuffle(_2d_anns_data)
        _2d_anns_data_train = _2d_anns_data[:int(len(_2d_anns_data) * (1 - validation_split))]
        _2d_anns_data_val = _2d_anns_data[int(len(_2d_anns_data) * (1 - validation_split)):]

        with open(csv_train_file_dir, 'w', encoding='UTF8', newline='') as csv_train_file:
            csv_train_file.truncate(0)
            csv_writer = csv.writer(csv_train_file)
            for annotation in tqdm(_2d_anns_data_train):
                csv_row = get_csv_row(nusc, annotation, render_on_screen=False)
                csv_writer.writerow(csv_row)
            print(f"Saved the formatted train dataset to {csv_train_file_dir}")

        with open(csv_val_file_dir, 'w', encoding='UTF8', newline='') as csv_val_file:
            csv_val_file.truncate(0)
            csv_writer = csv.writer(csv_val_file)
            for annotation in tqdm(_2d_anns_data_val):
                csv_row = get_csv_row(nusc, annotation, render_on_screen=False)
                csv_writer.writerow(csv_row)
            print(f"Saved the formatted validation dataset to {csv_val_file_dir}")


def get_csv_row(nusc, annotation, render_on_screen=False):
    """
    For a given annotation from the JSON 2D annotations file (generated through export_2d_annotations_as_json.py),
    this function returns a list of values which correspond to a row in the final dataset CSV file.

    Args:
        nusc: NuScenes object of the dataset
        annotation: annotation dictionary from the JSON 2D annotations file
        render_on_screen: set as True if you wish to render the annotations on screen as the dataset is being generated

    Returns: List of the values per row, with the following format: [image_path, bbox_x1, bbox_y1,
        bbox_x2, bbox_y2, class_id] (CSV writer separates them with commas automatically).
    """
    fused_imgs_dir = config['fused_imgs_dir']

    sample_data_token = annotation['sample_data_token']
    sample_data = nusc.get('sample_data', sample_data_token)
    sample_token = sample_data['sample_token']

    # get image
    image_path = os.path.join(fused_imgs_dir, sample_token + '.png')

    # get bounding boxes and class ids
    bounding_box = annotation['bbox_corners']
    bbox_x1 = int(bounding_box[0])
    bbox_y1 = int(bounding_box[1])
    bbox_x2 = int(bounding_box[2])
    bbox_y2 = int(bounding_box[3])

    class_id = annotation['category_name']

    if render_on_screen:
        new_img = cv2.imread(image_path)
        bbox_width = bbox_x2 - bbox_x1
        bbox_height = bbox_y2 - bbox_y1
        new_img = draw_bbox(new_img, bbox_x1, bbox_y1, bbox_width, bbox_height, class_id=class_id)
        cv2.imshow('Annotation render', new_img)
        cv2.waitKey(1)

    return [image_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id]


def generate_encoding_csv():
    """

    Generates a CSV file with the dataset encoding (animal = 0, human.pedestrian.adult = 1, ...). As per RetinaNet
    instructions, the format of each row is the following:

    > class_name,id

    """
    nuscenes_classes = [
        'animal',
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer',
        'human.pedestrian.stroller',
        'human.pedestrian.wheelchair',
        'movable_object.barrier',
        'movable_object.debris',
        'movable_object.pushable_pullable',
        'movable_object.trafficcone',
        'static_object.bicycle_rack',
        'vehicle.bicycle',
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.car',
        'vehicle.construction',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police',
        'vehicle.motorcycle',
        'vehicle.trailer',
        'vehicle.truck'
    ]

    dataset_version = config['dataset_version']
    if dataset_version != 'v1.0-mini' and dataset_version != 'v1.0-trainval':
        raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
    data_dir = config['data_dir']
    dataset_version_dir = os.path.join(data_dir, dataset_version)
    print("Generating the dataset CSV for dataset version %s:" % dataset_version)

    # check if the CSV file is already there
    csv_file_dir = os.path.join(dataset_version_dir, 'dataset_encoding.csv')
    if os.path.exists(csv_file_dir):
        print("======")
        print("Dataset encoding CSV file already exists at:")
        print(csv_file_dir)
        print("======")
        return 0

    with open(csv_file_dir, 'w', encoding='UTF8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for id, class_name in enumerate(nuscenes_classes):
            csv_row = [class_name, id]
            csv_writer.writerow(csv_row)

    print("Saved the dataset encoding to %s." % csv_file_dir)


if __name__ == '__main__':
    generate_dataset_csv(size_threshold=3, shuffle_data=True, validation_split=0.2, overwrite=True)
    generate_encoding_csv()
