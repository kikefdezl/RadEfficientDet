import os

config = {
    # Set up the name of the folder where the images with the fused radar data will be saved to.
    'fused_imgs_dir': os.path.join(os.environ.get('NUSCENES_DIR'), 'fused_imgs'),

    # set as 'v1.0-trainval' or 'v1.0-mini' (mini for testing purposes, trainval for the whole dataset)
    'dataset_version': 'v1.0-trainval',

    #### FUSION.PY ####
    # to show the fused data when running fusion.py, set as True. If False, the script saves the images as png files
    # instead.
    'fusion_show_images': True,
    'fusion_side': 'FRONT',  # set as FRONT, FRONT_RIGHT, FRONT_LEFT, BACK_RIGHT or BACK_LEFT. FRONT recommended: as of
    # now, lateral data can be fused but is unsupported for loading the dataset and training the network
    'fusion_hz': 0,  # number of milliseconds between frames when showing the fused data. Set as 0 for 'press key to
    # advance'. Press ESC to exit.

    # Must set up an environment variable 'NUSCENES_DIR' in your OS with the directory of your NuScenes database
    # e.g. NUSCENES_DIR = C:/Data/NuScenes
    'data_dir': os.environ.get('NUSCENES_DIR'),
}
