import cv2
import skimage.measure
import numpy as np


def pool3D(arr,
           kernel=(2, 2, 1),
           stride=(2, 2, 1),
           func=np.nanmax,
           ):
    # check inputs
    assert arr.ndim == 3
    assert len(kernel) == 3

    # create array with lots of padding around it, from which we grab stuff (could be more efficient, yes)
    arr_padded_shape = arr.shape + 2 * np.array(kernel)
    arr_padded = np.zeros(arr_padded_shape, dtype=arr.dtype) * np.nan
    arr_padded[
    kernel[0]:kernel[0] + arr.shape[0],
    kernel[1]:kernel[1] + arr.shape[1],
    kernel[2]:kernel[2] + arr.shape[2],
    ] = arr

    # create temporary array, which aggregates kernel elements in last axis
    size_x = 1 + (arr.shape[0] - 1) // stride[0]
    size_y = 1 + (arr.shape[1] - 1) // stride[1]
    size_z = 1 + (arr.shape[2] - 1) // stride[2]
    size_kernel = np.prod(kernel)
    arr_tmp = np.empty((size_x, size_y, size_z, size_kernel), dtype=arr.dtype)

    # fill temporary array
    kx_center = (kernel[0] - 1) // 2
    ky_center = (kernel[1] - 1) // 2
    kz_center = (kernel[2] - 1) // 2
    idx_kernel = 0
    for kx in range(kernel[0]):
        dx = kernel[0] + kx - kx_center
        for ky in range(kernel[1]):
            dy = kernel[1] + ky - ky_center
            for kz in range(kernel[2]):
                dz = kernel[2] + kz - kz_center
                arr_tmp[:, :, :, idx_kernel] = arr_padded[
                                               dx:dx + arr.shape[0]:stride[0],
                                               dy:dy + arr.shape[1]:stride[1],
                                               dz:dz + arr.shape[2]:stride[2],
                                               ]
                idx_kernel += 1

    # perform pool function
    arr_final = func(arr_tmp, axis=-1)
    return arr_final


def main1():
    file = 3
    filename = f"C:/Users/Kike/Desktop/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460_radar_P{str(file)}.jpg"

    n_pools = 7 - file

    image = cv2.imread(filename)
    cv2.imshow('original_image', image)
    pooled = image.copy()
    for pooln in range(n_pools):
        pooled = pool3D(pooled)
        cv2.imshow(f'pooledimage_{pooln}', pooled)
    cv2.waitKey(0)


def main2():
    filename = "C:/Users/Kike/Desktop/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"
    radar_filenames = [
        [f"C:/Users/Kike/Desktop/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460_radar_P{str(file + 3)}.jpg",
         7 - file] for file in range(5)]

    image = cv2.imread(filename)
    for radar_filename, layer in radar_filenames:
        pooled = cv2.imread(radar_filename)
        for pooln in range(layer-3):
            pooled = pool3D(pooled)
        cv2.imshow(f'P{layer}', pooled)
    cv2.waitKey(0)


if __name__ == '__main__':
    main2()
