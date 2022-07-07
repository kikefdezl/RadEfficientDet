import cv2
from tqdm import tqdm
import os

def main():
    camera_images_dir = "/mnt/TFM_KIKE/INFERENCES/cam_inf_on_rain/"
    radar_images_dir = "/mnt/TFM_KIKE/INFERENCES/rad_inf_on_rain_concat/"
    save_dir = "/mnt/TFM_KIKE/INFERENCES/inf_merged/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for image in tqdm(os.listdir(radar_images_dir)):
        image_cam_path = os.path.join(camera_images_dir, image)
        image_radar_path = os.path.join(radar_images_dir, image)

        image_cam = cv2.imread(image_cam_path)
        image_radar = cv2.imread(image_radar_path)

        merged_image = cv2.hconcat([image_cam, image_radar])

        cv2.imwrite(os.path.join(save_dir, image), merged_image)

if __name__ == "__main__":
    main()