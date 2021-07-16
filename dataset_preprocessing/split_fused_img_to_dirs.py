"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

"""
# standard libraries
import os
import sys
from shutil import copyfile, rmtree
import glob

# local
from config import config

# 3rd party libraries
from tqdm import tqdm


def split_to_dirs(n_dirs=3):

    data_dir = config['data_dir']
    fused_imgs_dir = config['fused_imgs_dir']

    # check that the fused imgs dataset exists
    if not os.path.exists(fused_imgs_dir):
        raise Exception("fused_imgs directory does not exist. You must generate it first with the fusion.py script.")

    # delete the existing split directories
    existing_dirs = glob.glob(os.path.join(data_dir, "fused_imgs_part_*"))
    for existing_dir in existing_dirs:
        rmtree(existing_dir)

    # save the file names into a list
    file_names = os.listdir(fused_imgs_dir)
    array_len = int(len(file_names)/n_dirs)

    # split the file names into 'n_dirs' arrays of equal size
    split_files = [file_names[i:i+array_len] for i in range(0, len(file_names), array_len)]
    if len(split_files) > n_dirs:
        split_files[n_dirs-1].append(split_files[-1])
        split_files.pop(-1)

    # copy the files to the new directories
    for i in tqdm(range(n_dirs)):
        part_dir = os.path.join(data_dir, f"fused_imgs_part_{i}")
        os.mkdir(part_dir)
        for file in split_files[i]:
            old_dir = os.path.join(fused_imgs_dir, file)
            new_dir = os.path.join(part_dir, file)
            copyfile(old_dir, new_dir)


if __name__ == "__main__":
    split_to_dirs(n_dirs=3)