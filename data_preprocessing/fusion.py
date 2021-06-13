# default libraries
import os

# local libraries
from image_graphics import draw_circles
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import RadarPointCloud

# 3rd party libraries
import cv2

data_dir = os.environ.get('NUSCENES_DIR')


def get_list_of_samples(nusc):
    """
    Args:
        nusc: NuScenes object of the database

    Returns:
        all_samples: a list containing all of the samples in the database (full samples, not just the token)
    """

    all_samples = []

    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        curr_sample = nusc.get('sample', first_sample_token)

        for _ in range(scene['nbr_samples'] - 1):
            all_samples.append(curr_sample)
            next_token = curr_sample['next']
            curr_sample = nusc.get('sample', next_token)
        all_samples.append(curr_sample)  # this appends the last sample of the scene

    return all_samples


def fuse_data(nusc, sample):
    """
    Args:
        nusc: NuScenes object of the database
        sample: single Sample object of the database

    Returns:
        fused_image: image containing the radar data fused into the camera image.
    """

    nusc_explorer = NuScenesExplorer(nusc)
    sample_token = sample['token']
    cam_front_token = sample['data']['CAM_FRONT']
    cam_front_data = nusc.get('sample_data', cam_front_token)
    cam_front_filename = cam_front_data['filename']
    cam_front_filename = os.path.join(data_dir, cam_front_filename)
    image = cv2.imread(cam_front_filename)

    radar_front_token = sample['data']['RADAR_FRONT']
    radar_front_data = nusc.get('sample_data', radar_front_token)
    radar_front_filename = radar_front_data['filename']
    radar_front_filename = os.path.join(data_dir, radar_front_filename)

    # FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0
    # vx_rms vy_rms
    radar_point_cloud = RadarPointCloud.from_file(radar_front_filename)
    radar_image = radar_point_cloud.points
    points, coloring, im = nusc_explorer.map_pointcloud_to_image(radar_front_token, cam_front_token)

    image = draw_circles(image, points)
    cv2.imshow('windowname', image)
    cv2.waitKey(50)



if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
    all_samples = get_list_of_samples(nusc)
    for sample in all_samples:
        fuse_data(nusc, sample)
