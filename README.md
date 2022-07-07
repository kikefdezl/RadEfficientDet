# RadCam-Net
A Deep Learning architecture for obstacle detection fusing Camera and Radar data.

![RadEfficientDet Diagram](images/radefficientdet_diagram_v4.jpg?raw=true "Title")

## Requirements:
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

To install all dependencies automatically:
`pip install -r requirements.txt`


# 1.- Generate fused data

The first step is to generate images that include the radar data. To do so, we must access all the camera images in the nuscenes dataset, project the radar data onto them, and save them to a directory.

### 1.1- Set-up environment variables

For the scripts to find the NuScenes database on your computer, it is necessary to set up an environment variable in
your OS called 'NUSCENES_DIR' pointing to the NuScenes dataset folder location.

`e.g.: NUSCENES_DIR = C:/Data/NuScenes`

### 1.2- Run dataset_preprocessing/fusion.py

Inside the config.py file, make sure you change the show_images (True / False) and dataset_version ('mini' / 'trainval') variables. This allows you to
select whether to view or save the images, as well as whether to use the sample dataset or the complete one.

`e.g.: 
fusion_show_images = False;
dataset_version = 'v1.0-trainval'
`

This example would save the images of the complete dataset.

### 1.3- Generate JSON file with the 2D bounding boxes

By default, the NuScenes dataset provides 3D bounding boxes of the objects. However, RetinaNet uses 2D bounding boxes for object
detection, so a bounding box projection must be performed. To do so, the NuScenes python-SDK provides a useful file:
[export_2d_annotations_as_json.py](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py). Run this file to generate
the JSON file with the 2D bounding box coordinates.

`$python -m nn_models.preprocessing.export_2d_annotations_as_json`

Remember to set the dataset version to v1.0-mini or v1.0-trainval to export the annotations for the sample or complete dataset accordingly.

### 1.4- Generate the dataset CSV file for RetinaNet

Retina has a built-in dataset generator to import the images and annotations automatically. CSV files with the annotations and the label encoding must be generated the following way:

`$python -m nn_models.preprocessing.generate_dataset_csv`

As before, this can be done for the v1.0-mini or v1.0-trainval datasets.