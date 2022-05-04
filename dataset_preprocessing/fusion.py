"""

Author: Enrique Fernández-Laguilhoat Sánchez-Biezma

Some parts of code have been taken from the NuScenes SDK and modified for this use case.

"""

# default libraries
import os
import shutil

# local libraries
from image_graphics import draw_overlay, draw_overlay_v2, draw_radar_maps, draw_radar_maps_v2
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
import yaml

# 3rd party libraries
import cv2
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm


class Fuser:
    def __init__(self, nusc, config={}):
        """
        Initializes a data fuser for the NuScenes database
        """
        self.cfg = config
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

    def _get_sample_data(self, sample, min_dist):
        side = self.cfg['FUSION']['sensor_side']
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
        cam_filename = os.path.join(self.cfg['nuscenes_dir'], cam_filename)
        image = cv2.imread(cam_filename)

        # radar
        radar_data = self.nusc.get('sample_data', radar_data_token)
        radar_filename = radar_data['filename']
        radar_filename = os.path.join(self.cfg['nuscenes_dir'], radar_filename)

        # extracting the radar information from the file. The RadarPointCloud.from_file function returns the following:
        # FIELDS
        # x
        # y
        # z
        # dyn_prop
        # id
        # rcs
        # vx
        # vy
        # vx_comp
        # vy_comp
        # is_quality_valid
        # ambig_state
        # x_rms
        # y_rms
        # invalid_state
        # pdh0
        # vx_rms
        # vy_rms
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
        mask = np.logical_and(mask, depths > min_dist)  # get only detections with more than min distance
        # filter the detections with y coordinate out of the image
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1)
        # filter the detections with x coordinate out of the image
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1)

        points = points[:, mask]
        depths = depths[mask]
        velocities = radar_point_cloud.points[8:10, mask]

        return image, points, depths, velocities, cam_filename

    def overlay_radar_data(self, sample_token, min_dist: float = 1.0):
        """
        Args:
            sample_token: single Sample token of the database

        Returns:
            fused_image: image containing the radar data fused into the camera image.
        """
        # access all the necessary sample data
        sample = self.nusc.get('sample', sample_token)

        image, points, depths, velocities, camera_filename = self._get_sample_data(sample, min_dist)
        camera_filename = os.path.basename(camera_filename)

        fused_img = draw_overlay(image, points, depths, velocities)

        return fused_img, camera_filename

    def overlay_radar_data_v2(self, sample_token, min_dist: float = 1.0):
        """
        Args:
            sample_token: single Sample token of the database

        Returns:
            fused_image: image containing the radar data fused into the camera image.
        """
        # access all the necessary sample data
        sample = self.nusc.get('sample', sample_token)

        image, points, depths, velocities, camera_filename = self._get_sample_data(sample, min_dist)
        camera_filename = os.path.basename(camera_filename)

        fused_img = draw_overlay_v2(image, points, depths, velocities)

        return fused_img, camera_filename


    def create_radar_maps(self, sample_token, min_dist: float = 1.0):
        sample = self.nusc.get('sample', sample_token)

        image, points, depths, velocities, camera_filename = self._get_sample_data(sample, min_dist)
        camera_filename = os.path.basename(camera_filename)

        radar_maps = draw_radar_maps(image, points, depths, velocities, n_layers=5)

        return image, radar_maps, camera_filename

    def create_radar_maps_v2(self, sample_token, min_dist: float = 1.0, checkerboard_dir=None):
        sample = self.nusc.get('sample', sample_token)

        image, points, depths, velocities, camera_filename = self._get_sample_data(sample, min_dist)
        camera_filename = os.path.basename(camera_filename)

        radar_maps = draw_radar_maps_v2(image, points, depths, velocities, n_layers=5, checkerboard_path=checkerboard_dir)

        return image, radar_maps, camera_filename

    def get_sample_tokens(self):
        return self.list_of_sample_tokens

def main():
    with open('../config.yaml') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    dataset_version = cfg['dataset_version']
    if dataset_version != 'v1.0-mini' and dataset_version != 'v1.0-trainval':
        raise Exception("The specified dataset version does not exist. Select 'mini' or 'trainval'.")
    nuscenes_dir = cfg['nuscenes_dir']

    nusc = NuScenes(version=dataset_version, dataroot=nuscenes_dir, verbose=True)

    fuser = Fuser(nusc, config=cfg)
    dataset_save_dir = cfg['dataset_save_dir']

    # loop through all the samples to fuse their data.
    show_imgs = cfg['FUSION']['show_images']
    # if show_imgs = True, render the images on screen
    if show_imgs:
        fusion_hz = cfg['FUSION']['fusion_hz']
        for sample_token in tqdm(fuser.get_sample_tokens()):
            fused_image = fuser.overlay_radar_data(sample_token)
            cv2.imshow('window_name', fused_image)
            key = cv2.waitKey(fusion_hz)
            if key == 27:
                break
    # if show_imgs = False, save the files to the fused_imgs dir
    else:
        saved_imgs_dir = os.path.join(dataset_save_dir, 'imgs')
        if not os.path.exists(saved_imgs_dir):
            os.mkdir(saved_imgs_dir)
        else:
            shutil.rmtree(saved_imgs_dir)
            os.mkdir(saved_imgs_dir)
        for sample_token in tqdm(fuser.get_sample_tokens()):
            # saving the image
            image, radar_maps, camera_filename = fuser.create_radar_maps_v2(sample_token,
                                                    checkerboard_dir=cfg['FUSION']['checkerboard_1600x900_img_path'])
            full_path = os.path.join(saved_imgs_dir, camera_filename)
            cv2.imwrite(full_path, image)
            for idx, map in enumerate(radar_maps):
                map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
                full_path_map = os.path.splitext(full_path)[0] + f'_radar_P{7-idx}.jpg'
                cv2.imwrite(full_path_map, map)

        # for sample_token in tqdm(fuser.get_sample_tokens()):
        #     fused_image, camera_filename = fuser.overlay_radar_data_v2(sample_token)
        #     # saving the image
        #     full_path = os.path.join(saved_imgs_dir, os.path.basename(camera_filename))
        #     cv2.imwrite(full_path, fused_image)

if __name__ == "__main__":
    main()

