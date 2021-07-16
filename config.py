import os

config = {
    #### GENERAL PARAMETERS ####
    'data_dir': os.environ.get('NUSCENES_DIR'),  # DO NOT MODIFY. Must set up an environment variable 'NUSCENES_DIR' in
    # your OS with the directory of your NuScenes database: e.g. NUSCENES_DIR = C:/Data/NuScenes

    'fused_imgs_dir': os.path.join(os.environ.get('NUSCENES_DIR'), 'fused_imgs'),  # Set up the name of the folder where
    # the images with the fused radar data will be saved to.

    'dataset_version': 'v1.0-mini',  # set as 'v1.0-trainval' or 'v1.0-mini' (mini for testing purposes, trainval for
    # the whole dataset)


    #### FUSION.PY  PARAMETERS ####
    'fusion_show_images': True,  # to show the fused data when running fusion.py, set as True. If False, the script
    # saves the images as png files instead.

    'fusion_side': 'FRONT',  # set as FRONT, FRONT_RIGHT, FRONT_LEFT, BACK_RIGHT or BACK_LEFT. FRONT recommended: as of
    # now, lateral data can be fused but is unsupported for loading the dataset and training the network

    'fusion_hz': 0,  # number of milliseconds between frames when showing the fused data. Set as 0 for 'press key to
    # advance'. Press ESC to exit.


    #### SPLIT_FUSED_IMGS_TO_ZIPS.PY  PARAMETERS ####
    # set the number of zip files to divide the image dataset into
    'n_dirs': 7,


    #### GENERATE_DATASET_CSV.PY  PARAMETERS ####
    'size_threshold': 3,  # set the minimum bounding box height/width (in pixels). Smaller bboxes are deleted.
    'shuffle_data': True,  # set as True to shuffle the dataset CSV
    'validation_split': 0.2,  # percentage of the dataset to assign as validation data
    'overwrite': True  # set as True to overwrite existing CSV file. If False, an exception is returned if CSV exists.
}
