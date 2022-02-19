"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

This file is used to generate CSV files in the format: path/to/image.jpg,x1,y1,x2,y2,class_name

"""
# default libraries
import os
import json
import csv
from random import shuffle
from time import time
import sys

# 3rd party libraries
from nuscenes.nuscenes import NuScenes
import cv2
from tqdm import tqdm

# local libraries
sys.path.append("..")
from image_graphics import draw_bbox
import yaml


class CSVFileGenerator:
    def __init__(self, image_dir, size_threshold=3, shuffle_data=True, validation_split: float = 0.2, overwrite=False):
        self.size_threshold = size_threshold
        self.shuffle_data = shuffle_data
        self.validation_split = validation_split
        self.overwrite = overwrite

        self.dataset_version = config['dataset_version']
        self.data_dir = config['data_dir']
        self.fused_imgs_dir = image_dir

    def generate_dataset_csv(self):
        """
        Generates two csv files (training and validation) containing one annotation per row. The format of each row is:

        > path/to/image.jpg,x1,y1,x2,y2,class_name
        Args:
            overwrite       : set as True to overwrite files if they are already on disk. If not, the program will exit upon
                            finding the files
            validation_split: float value to determine the percentage of data to save to the validation CSV file
            shuffle_data    : set as true to shuffle all the parameters before creating the CSV file
            size_threshold  : bounding box height or width (in pixels), below which an annotation is deleted. Used for
                            filtering excessively small bounding boxes.

        """

        if self.dataset_version != 'v1.0-mini' and self.dataset_version != 'v1.0-trainval':
            raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
        dataset_version_dir = os.path.join(self.data_dir, self.dataset_version)

        # check if the CSV files are already there
        csv_train_file_dir = os.path.join(dataset_version_dir, 'train_dataset.csv')
        csv_val_file_dir = os.path.join(dataset_version_dir, 'val_dataset.csv')
        if not self.overwrite:
            if os.path.exists(csv_train_file_dir) and os.path.exists(csv_val_file_dir):
                print("======")
                print("Dataset CSV files already exist at:")
                print(csv_train_file_dir)
                print(csv_val_file_dir)
                print("======")
                return 0

        # check that the validation split is between 0 and 1
        assert 0.0 <= self.validation_split <= 1.0, "Incorrect validation_split value. It must be a float value between 0 & 1."

        nusc = NuScenes(version=self.dataset_version, dataroot=self.data_dir, verbose=True)

        # check that the 2D annotations file exists
        anns_dir = os.path.join(dataset_version_dir, 'image_annotations.json')
        if not os.path.exists(anns_dir):
            raise Exception(
                "No annotation data. Must generate 2D annotation JSON file first. Run '/dataset_preprocessing/export"
                "_2d_annotations_as_json.py'")

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
            if ((bbox_corners[2] - bbox_corners[0]) < self.size_threshold or
                    (bbox_corners[3] - bbox_corners[1]) < self.size_threshold):
                _2d_anns_data.pop(index)
                removed += 1
            else:
                index += 1
        print(f"Removed {removed} annotations with width or height below {self.size_threshold} pixels.")
        print("======")

        print("Grouping annotations per-image...")
        all_data = {}
        for annotation in tqdm(_2d_anns_data):
            filename = os.path.basename(annotation['filename'])
            all_data.setdefault(filename, {'filename': filename,
                                           'annotations': []})
            all_data[filename]['annotations'].append(annotation)
        print("Done.")

        # TODO: ADD IMAGES THAT DONT HAVE ANNOTATIONS


        # convert the dict to a list
        all_data = [value for (key, value) in all_data.items()]

        # shuffle and split into trainval
        shuffle(all_data)
        split_idx = int(len(all_data) * validation_split)
        data_val = all_data[:split_idx]
        data_train = all_data[split_idx:]

        print("Generating the dataset CSV for dataset version %s:" % self.dataset_version)
        with open(csv_train_file_dir, 'w', encoding='UTF8', newline='') as csv_train_file:
            csv_train_file.truncate(0)
            csv_writer = csv.writer(csv_train_file)
            for image in tqdm(data_train):
                for annotation in image['annotations']:
                    filename = os.path.join(self.fused_imgs_dir, image['filename'])
                    xmin = int(annotation['bbox_corners'][0])
                    ymin = int(annotation['bbox_corners'][1])
                    xmax = int(annotation['bbox_corners'][2])
                    ymax = int(annotation['bbox_corners'][3])
                    label = annotation['category_name']
                    csv_row = [filename, xmin, ymin, xmax, ymax, label]
                    csv_writer.writerow(csv_row)
        print(f"Saved the formatted train dataset to {csv_train_file_dir}")

        print("Generating the dataset CSV for dataset version %s:" % self.dataset_version)
        with open(csv_val_file_dir, 'w', encoding='UTF8', newline='') as csv_val_file:
            csv_val_file.truncate(0)
            csv_writer = csv.writer(csv_val_file)
            for image in tqdm(data_val):
                for annotation in image['annotations']:
                    filename = os.path.join(self.fused_imgs_dir, image['filename'])
                    xmin = int(annotation['bbox_corners'][0])
                    ymin = int(annotation['bbox_corners'][1])
                    xmax = int(annotation['bbox_corners'][2])
                    ymax = int(annotation['bbox_corners'][3])
                    label = annotation['category_name']
                    csv_row = [filename, xmin, ymin, xmax, ymax, label]
                    csv_writer.writerow(csv_row)
        print(f"Saved the formatted val dataset to {csv_val_file_dir}")


    def generate_encoding_csv(self):
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

        if self.dataset_version != 'v1.0-mini' and self.dataset_version != 'v1.0-trainval':
            raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
        dataset_version_dir = os.path.join(self.data_dir, self.dataset_version)
        print("Generating the dataset encoding CSV for dataset version %s:" % self.dataset_version)

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
    with open('../config.yaml') as yf:
        config = yaml.safe_load(yf)

    image_dir = config['dataset_save_dir']
    size_threshold = config['generate_dataset_CSV']['size_threshold']
    shuffle_data = config['generate_dataset_CSV']['shuffle_data']
    validation_split = config['generate_dataset_CSV']['validation_split']
    overwrite = config['generate_dataset_CSV']['overwrite']

    generator = CSVFileGenerator(image_dir=image_dir,
                                 size_threshold=size_threshold,
                                 shuffle_data=shuffle_data,
                                 validation_split=validation_split,
                                 overwrite=overwrite)

    generator.generate_dataset_csv()
    generator.generate_encoding_csv()
