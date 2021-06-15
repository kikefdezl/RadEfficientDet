# default libraries
import os

# local libraries
from image_graphics import draw_overlay
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import RadarPointCloud

# 3rd party libraries
import cv2
import numpy as np

data_dir = os.environ.get('NUSCENES_DIR')


class Fuser:

    def __init__(self, nusc):
        """
        initializes a data fuser with the NuScenes database
        """
        self.nusc = nusc
        self.nusc_explorer = NuScenesExplorer(self.nusc)

    def get_list_of_samples(self):
        """
        Returns: list_of_samples: a list containing all of the samples' tokens in the database (only the token)
        """

        list_of_samples = []

        for scene in self.nusc.scene:
            first_sample_token = scene['first_sample_token']
            curr_sample = self.nusc.get('sample', first_sample_token)

            for _ in range(scene['nbr_samples'] - 1):
                list_of_samples.append(curr_sample['token'])
                next_token = curr_sample['next']
                curr_sample = self.nusc.get('sample', next_token)
            list_of_samples.append(curr_sample['token'])  # this appends the last sample of the scene

        return list_of_samples

    def fuse_data(self, sample_token, min_dist: float = 1.0, side: str = 'FRONT'):
        """
        Args:
            sample_token: single Sample object of the database

        Returns:
            fused_image: image containing the radar data fused into the camera image.
        """
        # access all the necessary sample data
        sample = self.nusc.get('sample', sample_token)

        if side == 'FRONT':
            cam_token = sample['data']['CAM_FRONT']
            radar_token = sample['data']['RADAR_FRONT']
        elif side == 'FRONT_RIGHT':
            cam_token = sample['data']['CAM_FRONT_RIGHT']
            radar_token = sample['data']['RADAR_FRONT_RIGHT']
        elif side == 'FRONT_LEFT':
            cam_token = sample['data']['CAM_FRONT_LEFT']
            radar_token = sample['data']['RADAR_FRONT_LEFT']

        # camera
        cam_front_data = self.nusc.get('sample_data', cam_token)
        cam_front_filename = cam_front_data['filename']
        cam_front_filename = os.path.join(data_dir, cam_front_filename)
        image = cv2.imread(cam_front_filename)

        # radar
        radar_front_data = self.nusc.get('sample_data', radar_token)
        radar_front_filename = radar_front_data['filename']
        radar_front_filename = os.path.join(data_dir, radar_front_filename)

        # extracting the radar information from the file. The RadarPointCloud.from_file function returns the following:
        # FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0
        # vx_rms vy_rms
        radar_point_cloud = RadarPointCloud.from_file(radar_front_filename)
        points = radar_point_cloud.points
        depths = radar_point_cloud.points[2, :]

        """
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < image.shape[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < image.shape[1] - 1)
        points = points[:, mask]
        """

        # the map_pointcloud_to_image function does the necessary transformations to overlay the radar points on top
        # of the corresponding image. It also exports the different coloring depending on point depth.
        mapped_points, coloring, im = self.nusc_explorer.map_pointcloud_to_image(radar_token, cam_token)

        fused_img = draw_overlay(image, mapped_points, coloring, points)

        cv2.imshow('window_name', image)
        cv2.waitKey()

        return fused_img


if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    fuser = Fuser(nusc)
    list_of_samples = fuser.get_list_of_samples()
    for sample_token in list_of_samples:
        fused_image = fuser.fuse_data(sample_token)
