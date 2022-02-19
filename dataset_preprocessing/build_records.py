"""
Usage:
  # From the data set dir
  # Create train data:
  python ../generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util
from tqdm import tqdm
from time import time

from collections import namedtuple

import records_config as config


def generate_class_file(classes, classes_file):
    # open the classes output file
    with open(classes_file, "w") as f:
        # loop over the classes
        for (k, v) in classes.items():
            # construct the class information and write to file
            item = ("item {\n"
                    "\tid: " + str(v) + "\n"
                                        "\tname: '" + k + "'\n"
                                                          "}\n")
            f.write(item)


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group):
    filename = group.filename

    encoded = tf.io.gfile.GFile(filename, "rb").read()
    encoded = bytes(encoded)


    image = cv2.imread(filename)
    image_format = b'jpg'
    height, width = image.shape[:2]

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        class_id = config.CLASSES[row['class']]
        classes.append(class_id)

    filename = filename.encode('utf8')
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    generate_class_file(config.CLASSES, config.CLASSES_FILE)

    time_start = time()
    print(f"Creating train.record at {config.TRAIN_RECORD}")
    with tf.io.TFRecordWriter(config.TRAIN_RECORD) as writer_train:
        examples_train = pd.read_csv(config.TRAIN_ANNOT_PATH, names=["filename", "xmin", "ymin", "xmax", "ymax", "class"])
        grouped_train = split(examples_train, 'filename')

        for group in tqdm(grouped_train):
            tf_example = create_tf_example(group)
            writer_train.write(tf_example.SerializeToString())
    print(f"Done in {(time() - time_start):.2f} seconds.")

    time_start = time()
    print(f"Creating val.record at {config.VAL_RECORD}")
    with tf.io.TFRecordWriter(config.VAL_RECORD) as writer_val:
        examples_val = pd.read_csv(config.VAL_ANNOT_PATH, names=["filename", "xmin", "ymin", "xmax", "ymax", "class"])
        grouped_val = split(examples_val, 'filename')

        for group in tqdm(grouped_val):
            tf_example = create_tf_example(group)
            writer_val.write(tf_example.SerializeToString())
    print(f"Done in {(time() - time_start):.2f} seconds.")


if __name__ == '__main__':
    tf.compat.v1.app.run()