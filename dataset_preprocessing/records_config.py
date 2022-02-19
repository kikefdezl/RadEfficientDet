#!/usr/bin/env python3
'''
Author:  Alberto Garcia Perez
Subject: Tensorflow Object Detector preprocess configuration
Date:    2020-01
Usage:   imported by buildRecords.py
Links:
'''

# import the necessary packages
import os

# initialize the base path for the LISA dataset
BASE_PATH    = os.path.join(os.path.sep,'/mnt/TFM_KIKE/DATASETS/fused_imgs_no_visibility_0_real/')
VAL_ANNOT_PATH   = os.path.sep.join([BASE_PATH,'val.csv'])
TRAIN_ANNOT_PATH   = os.path.sep.join([BASE_PATH,'train.csv'])
TRAIN_RECORD = os.path.sep.join([BASE_PATH,'train.record'])
VAL_RECORD  = os.path.sep.join([BASE_PATH,'val.record'])
CLASSES_FILE = os.path.sep.join([BASE_PATH,'classes.pbtxt'])

# initialize the class labels dictionary
CLASSES = {'animal': 1,
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
            'vehicle.truck': 23,
           }

