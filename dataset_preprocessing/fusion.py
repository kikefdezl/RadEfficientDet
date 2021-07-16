"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

Some parts of code have been taken from the NuScenes SDK and modified for this use case.

"""

# default libraries
import os

# local libraries
from dataset_preprocessing.image_graphics import draw_overlay
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from config import config

# 3rd party libraries
import cv2
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm


class Fuser:
    def __init__(self, nusc):
        """
        Initializes a data fuser for the NuScenes database
        """
        self.nusc = nusc
        self.nusc_explorer = NuScenesExplorer(self.nusc)
        self.list_of_sample_tokens = []
        self._generate_sample_tokens()

    def _generate_sample_tokens(self):
        """
        generates a list of all the sample tokens in the dataset, and saves it in the self.list_of_sample_tokens
        variable
        """

        self.list_of_sample_tokens = []

        for scene in self.nusc.scene:
            first_sample_token = scene['first_sample_token']
            curr_sample = self.nusc.get('sample', first_sample_token)

            for _ in range(scene['nbr_samples'] - 1):
                self.list_of_sample_tokens.append(curr_sample['token'])
                next_token = curr_sample['next']
                curr_sample = self.nusc.get('sample', next_token)
            self.list_of_sample_tokens.append(curr_sample['token'])  # this appends the last sample token of the scene

    def fuse_data(self, sample_token, min_dist: float = 1.0):
        """
        Args:
            sample_token: single Sample token of the database

        Returns:
            fused_image: image containing the radar data fused into the camera image.
        """
        # access all the necessary sample data
        sample = self.nusc.get('sample', sample_token)

        side = config['fusion_side']
        if side == 'FRONT':
            cam_data_token = sample['data']['CAM_FRONT']
            radar_data_token = sample['data']['RADAR_FRONT']
        elif side == 'FRONT_RIGHT':
            cam_data_token = sample['data']['CAM_FRONT_RIGHT']
            radar_data_token = sample['data']['RADAR_FRONT_RIGHT']
        elif side == 'FRONT_LEFT':
            cam_data_token = sample['data']['CAM_FRONT_LEFT']
            radar_data_token = sample['data']['RADAR_FRONT_LEFT']
        elif side == 'BACK_RIGHT':
            cam_data_token = sample['data']['CAM_BACK_RIGHT']
            radar_data_token = sample['data']['RADAR_BACK_RIGHT']
        elif side == 'BACK_LEFT':
            cam_data_token = sample['data']['CAM_BACK_LEFT']
            radar_data_token = sample['data']['RADAR_BACK_LEFT']
        else:
            raise Exception("Error, Fuser side < %s > doesn't exist, choose one of the following: FRONT, FRONT_RIGHT, "
                            "FRONT_LEFT, BACK_RIGHT, BACK_LEFT" % side)

        # camera
        cam_data = self.nusc.get('sample_data', cam_data_token)
        cam_filename = cam_data['filename']
        cam_filename = os.path.join(config['data_dir'], cam_filename)
        image = cv2.imread(cam_filename)

        # radar
        radar_data = self.nusc.get('sample_data', radar_data_token)
        radar_filename = radar_data['filename']
        radar_filename = os.path.join(config['data_dir'], radar_filename)

        # extracting the radar information from the file. The RadarPointCloud.from_file function returns the following:
        # FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0
        # vx_rms vy_rms
        radar_point_cloud = RadarPointCloud.from_file(radar_filename)

        # most of the following code has been taken from nuscenes.py > NuScenesExplorer.map_pointcloud_to_img. It has
        # been modified to also provide velocity vector information

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
        radar_point_cloud.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        radar_point_cloud.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        pose_record = self.nusc.get('ego_pose', radar_data['ego_pose_token'])
        radar_point_cloud.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
        radar_point_cloud.translate(np.array(pose_record['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        pose_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        radar_point_cloud.translate(-np.array(pose_record['translation']))
        radar_point_cloud.rotate(Quaternion(pose_record['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        radar_point_cloud.translate(-np.array(cs_record['translation']))
        radar_point_cloud.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = radar_point_cloud.points[2, :]

        points = view_points(radar_point_cloud.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1)

        points = points[:, mask]
        depths = depths[mask]
        velocities = radar_point_cloud.points[8:10, mask]

        fused_img = draw_overlay(image, points, depths, velocities)

        return fused_img

    def get_sample_tokens(self):
        return self.list_of_sample_tokens


if __name__ == "__main__":

    dataset_version = config['dataset_version']
    if dataset_version != 'v1.0-mini' and dataset_version != 'v1.0-trainval':
        raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
    data_dir = config['data_dir']

    nusc = NuScenes(version=dataset_version, dataroot=config['data_dir'], verbose=True)

    fuser = Fuser(nusc)
    fused_imgs_dir = config['fused_imgs_dir']

    # loop through all the samples to fuse their data.
    show_imgs = config['fusion_show_images']
    # if show_imgs = True, render the images on screen
    if show_imgs:
        fusion_hz = config['fusion_hz']
        for sample_token in tqdm(fuser.get_sample_tokens()):
            fused_image = fuser.fuse_data(sample_token)
            cv2.imshow('window_name', fused_image)
            key = cv2.waitKey(fusion_hz)
            if key == 27:
                break
    # if show_imgs = False, save the files to the fused_imgs dir
    else:
        for sample_token in tqdm(fuser.get_sample_tokens()):
            fused_image = fuser.fuse_data(sample_token)
            # saving the image
            img_filename = str(sample_token) + '.png'
            img_filename = os.path.join(fused_imgs_dir, img_filename)
            cv2.imwrite(img_filename, fused_image)
