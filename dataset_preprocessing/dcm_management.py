""" THIS SCRIPT CONTAINS FUNCTIONS TO MANAGE DICOM FILES.
    FUNCTIONS:
    - loadDCM. This calls to:
        - isNumberOfFramesWrong
        - findImageBits
"""

# IMPORT MODULES
import sys
import os
import numpy as np
import pydicom


# THIS FUNCTION LOAD THE IMAGE SAVED IN A DICOM FILE INTO A PYTHON ARRAY (MATRIX)
def loadDCM(dicomImagePath, verbose=False):
    """ Loads a single File """

    try:
        dataset = pydicom.dcmread(dicomImagePath)

        # Images from PCB have wrong DICOM format
        # Number of Frames = 20 but only 1 frame available
        # Leads to error message, so Number of Frames modified herein:
        if dataset.NumberOfFrames != None:

            if isNumberOfFramesWrong(dataset):
                # print('[WARNING] ********  Changing number of Frames to 1 ***')
                dataset.NumberOfFrames = 1

            # Some dcm images (i.e. generated from ASTM-2660) have a strange
            # dtype (>u2), which leads to wrong visualization
            # this needs to be corrected for these particular cases
            img = dataset.pixel_array
            if img.dtype == '>u2':
                imageBit = findImageBits(img)
                img = np.uint16(img) if imageBit == 16 else np.uint8(img)

            if verbose:
                # Checking output is 16 bit depth image
                np.set_printoptions(threshold=np.inf)
                # print('[INFO] Dicomloader', img.dtype)
            return img, True
        else:
            return [], False

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]


def isNumberOfFramesWrong(dataset):
    """ Images from PCB have wrong DICOM format
    Number of Frames = 20 but only 1 frame available
    Leads to error message, so Number of Frames may need to be modified """

    try:
        if dataset.NumberOfFrames == '':
            return True
        if dataset.NumberOfFrames > 1:
            return True

    except Exception as ex:
        print("[INFO] Dicom image with ", ex)
        return False


def findImageBits(img):
    """
    Subject:    Finds the bits of an image. Only for 8, 16 bits and 12 bits (for scanned images)
    Author:     Alberto Garcia Perez
    Date:       2018/11
    Comments:   Return image bits
    """

    if img is None:
        return 0

    maxValue = img.max()

    if maxValue <= 255:
        return 8

    elif maxValue <= 4096:
        return 12

    else:
        return 16
