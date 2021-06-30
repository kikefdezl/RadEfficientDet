# RadCam-Net
A Deep Learning project for obstacle detection fusing Camera and Radar data.

# Requirements:
- Python 3.7
- [Nuscenes dataset](https://www.nuscenes.org/)
- Nuscenes-devkit `pip install nuscenes-devkit` 
- Tensorflow (+ Keras) `pip install tensorflow` `pip install tensorflow_datasets`
- Matplotlib `pip install matplotlib`
- OpenCV `pip install opencv_python`
- Sklearn `pip install sklearn`
- PyQuaternion `pip install pyquaternion`
- TQDM `pip install tqdm`
- Cache Tools `pip install cachetools`


# 1.- Generate fused data

The first step to this method is to generate images that include the radar data. To do so, we must access all the front camera images in the nuscenes dataset, project the radar data onto them, and save them to a directory.

### 1.1.- Set-up environment variables

For the scripts to find the NuScenes database on your computer, it is necessary to set up an environment variable in
your OS called 'NUSCENES_DIR' pointing to the NuScenes folder location.

`e.g.: NUSCENES_DIR = C:/Data/NuScenes`

### 1.2.- Run data_preprocessing/fusion.py

Inside the fusion.py file, make sure you change the show_images (True / False) and dataset_version ('mini' / 'trainval') variables in the head of the code. This allows you to
select whether to view or save the images, as well as whether to use the sample dataset or the complete one.

`e.g.: 
show_images = False;
dataset_version = 'trainval;'
`

This example would save the images of the complete dataset.

