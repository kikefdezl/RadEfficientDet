"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

This script was created to divide the fused_imgs directory (~50 GB) into n separate ZIP files. This makes it easier
to manage the files for certain applications, like uploading the files to Google Drive for use with Google Colab.

For this script to work, you must have generated the fused data beforehand, through the fusion.py script. The fused
images produced by fusion.py are saved at the NuScenes directory (that has been set as an environment variable):
('path_to_NuScenes/fused_imgs/')

This script generates the new directories and ZIP files in the same path, with the format:
dir: ('path_to_NuScenes/fused_imgs_part_*/NuScenes/fused_imgs/')
zip: ('path_to_NuScenes/fused_imgs_part_*.zip)

The annotations are also zipped into 'path_to_NuScenes/fused_imgs_metadata.zip')
"""
# standard libraries
import os
from shutil import copyfile, rmtree, make_archive, copytree
import glob

# local
from config import config

# 3rd party libraries
from tqdm import tqdm


def zip_metadata():
    data_dir = config['data_dir']

    # remove the existing metadata tree and ZIP file
    rmtree(os.path.join(data_dir, 'fused_imgs_metadata'), ignore_errors=True)
    try:
        os.remove(os.path.join(data_dir, 'fused_imgs_metadata.zip'))
    except OSError:
        pass

    print("Copying the metadata files to new directory...")
    metadata_copy_dir = os.path.join(data_dir, 'fused_imgs_metadata', 'NuScenes')
    os.makedirs(metadata_copy_dir, exist_ok=True)
    copytree(os.path.join(data_dir, 'v1.0-mini'), os.path.join(metadata_copy_dir, 'v1.0-mini'))
    copytree(os.path.join(data_dir, 'v1.0-trainval'), os.path.join(metadata_copy_dir, 'v1.0-trainval'))

    print("Creating metadata ZIP file...")
    make_archive(os.path.join(data_dir, 'fused_imgs_metadata'), 'zip', root_dir=os.path.join(data_dir,
                                                                                             'fused_imgs_metadata'))


def split_imgs_to_zips(n_dirs=7):
    data_dir = config['data_dir']
    fused_imgs_dir = config['fused_imgs_dir']

    # check that the fused imgs dataset exists
    if not os.path.exists(fused_imgs_dir):
        raise Exception("fused_imgs directory does not exist. You must generate it first with the fusion.py script.")

    # delete the existing split directories and zip files
    existing_dirs = glob.glob(os.path.join(data_dir, "fused_imgs_part_*/"))
    existing_zips = glob.glob(os.path.join(data_dir, "fused_imgs_part_*.zip"))
    for existing_dir in existing_dirs:
        rmtree(existing_dir)
    for existing_zip in existing_zips:
        os.remove(existing_zip)

    # save the file names into a list
    file_names = os.listdir(fused_imgs_dir)
    array_len = int(len(file_names) / n_dirs)

    # split the file names into 'n_dirs' arrays of equal size
    split_files = [file_names[i:i + array_len] for i in range(0, len(file_names), array_len)]
    if len(split_files) > n_dirs:
        split_files[n_dirs - 1].extend(split_files[-1])
        split_files.pop(-1)

    # copy the files to the new directories
    for i in range(n_dirs):
        part_dir = os.path.join(data_dir, f'fused_imgs_part_{i}')
        save_dir = os.path.join(part_dir, 'NuScenes', 'fused_imgs')
        os.makedirs(save_dir)
        print(f"Part {i}/{n_dirs - 1}: Copying the image files to new directory...")
        for file in tqdm(split_files[i], leave=False):
            old_dir = os.path.join(fused_imgs_dir, file)
            new_dir = os.path.join(save_dir, file)
            copyfile(old_dir, new_dir)

        print(f"Part {i}/{n_dirs - 1}: Creating ZIP file...")
        make_archive(os.path.join(data_dir, f'fused_imgs_part_{i}'), 'zip', root_dir=part_dir)


if __name__ == "__main__":
    n_dirs = config['n_dirs']

    zip_metadata()
    split_imgs_to_zips(n_dirs=n_dirs)
