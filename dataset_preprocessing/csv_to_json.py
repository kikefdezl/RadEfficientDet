import json
import os
from tqdm import tqdm
from random import shuffle
import pandas as pd

def main():
    dataset_dir = '/mnt/TFM_KIKE/DATASETS/fused_imgs_v2'
    csv_train_file = '/mnt/TFM_KIKE/DATASETS/fused_imgs_v2/train.csv'
    csv_val_file = '/mnt/TFM_KIKE/DATASETS/fused_imgs_v2/val.csv'

    json_2d_anns = '/mnt/TFM_KIKE/DATASETS/fused_imgs_v2/image_annotations.json'

    with open(json_2d_anns) as json_file:
        anns = json.load(json_file)

    header = ['filename', 'x1', 'y1', 'x2', 'y2', 'class']
    train_dataframe = pd.read_csv(csv_train_file, names=header)
    val_dataframe = pd.read_csv(csv_val_file, names=header)

    train_filenames = dict.fromkeys(set([os.path.basename(row) for row in train_dataframe['filename']]), [])
    val_filenames = dict.fromkeys(set([os.path.basename(row) for row in val_dataframe['filename']]), [])

    data_train = []
    data_val = []

    n=0
    for annotation in tqdm(anns):
        filename = os.path.basename(annotation['filename'])
        if filename in train_filenames.keys():
            train_filenames[filename].append(annotation)
        elif filename in val_filenames.keys():
            val_filenames[filename].append(annotation)
        else:
            n+=1
            print(f"{n}{filename} not in train or val lists")

    pass

def main2():
    json_2d_anns = '/mnt/TFM_KIKE/DATASETS/fused_imgs_v2/image_annotations.json'
    train_json = '/mnt/TFM_KIKE/DATASETS/fused_imgs_v2/train.json'
    val_json = '/mnt/TFM_KIKE/DATASETS/fused_imgs_v2/val.json'
    test_json = '/mnt/TFM_KIKE/DATASETS/fused_imgs_v2/test.json'

    with open(json_2d_anns) as json_file:
        anns = json.load(json_file)

    filenames = [os.path.basename(annotation['filename']) for annotation in anns]

    rem_dupl_filenames = list(set(filenames))

    shuffle(rem_dupl_filenames)

    index_train = int(0.8 * len(rem_dupl_filenames))
    index_val = index_train + int(0.1 * len(rem_dupl_filenames))

    train_filenames = set(rem_dupl_filenames[:index_train])
    val_filenames = set(rem_dupl_filenames[index_train:index_val])
    test_filenames = set(rem_dupl_filenames[index_val:])

    train_dict = {key:[] for key in train_filenames}
    val_dict = {key:[] for key in val_filenames}
    test_dict = {key:[] for key in test_filenames}

    for annotation in tqdm(anns):
        filename = os.path.basename(annotation['filename'])
        if filename in train_dict:
            train_dict[filename].append(annotation)
        elif filename in val_dict:
            val_dict[filename].append(annotation)
        elif filename in test_dict:
            test_dict[filename].append(annotation)
        else:
            print(f"{filename} not in train or val lists")

    train_data = [{'filename': key,
                   'annotations': train_dict[key]} for key in train_dict]
    val_data = [{'filename': key,
                   'annotations': val_dict[key]} for key in val_dict]
    test_data = [{'filename': key,
                   'annotations': test_dict[key]} for key in test_dict]

    with open(train_json, 'w') as jf:
        json.dump(train_data, jf, indent="    ")
    with open(val_json, 'w') as jf:
        json.dump(val_data, jf, indent="    ")
    with open(test_json, 'w') as jf:
        json.dump(test_data, jf, indent="    ")


if __name__ == '__main__':
    main2()