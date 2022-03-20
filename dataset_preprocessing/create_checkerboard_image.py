import numpy as np
import cv2

def main():
    image_height = 900
    image_width = 1600
    checker_size = 4
    save_path = "/mnt/TFM_KIKE/DATASETS/fused_imgs_v4_no_visibility_0/checkerboard1.png"

    image = np.zeros((image_height, image_width))

    for step_x in range(0, image_width, checker_size*2):
        for step_y in range(0, image_height, checker_size*2):
            image = cv2.rectangle(image, (step_x, step_y), (step_x + checker_size, step_y + checker_size), color=(255, 255, 255), thickness=-1)
    for step_x in range(checker_size, image_width, checker_size*2):
        for step_y in range(checker_size, image_height, checker_size*2):
            image = cv2.rectangle(image, (step_x, step_y), (step_x + checker_size, step_y + checker_size), color=(255, 255, 255), thickness=-1)

    cv2.imwrite(save_path, image)

if __name__ == "__main__":
    main()