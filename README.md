# RadCam-Net
A Deep Learning platform for obstacle detection fusing Camera and Radar data.

# Requirements:
- Python 3.7
- [Nuscenes dataset](https://www.nuscenes.org/)
- Tensorflow (+ Keras) `pip install tensorflow` `pip install tensorflow_datasets`
- Matplotlib `pip install matplotlib`
- OpenCV `pip install opencv_python`
- Sklearn `pip install sklearn`
- PyQuaternion `pip install pyquaternion`
- TQDM `pip install tqdm`
- Cache Tools `pip install cachetools`

# Set-up environment variables

For the scripts to find the NuScenes database on your computer, it is necessary to set up an environment variable in
your OS called 'NUSCENES_DIR' pointing to the NuScenes folder location.

`eg: NUSCENES_DIR = C:/Data/NuScenes`

