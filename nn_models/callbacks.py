"""
Author: Enrique Fernández-Laguilhoat Sánchez-Biezma
"""
# default libraries
import os
from shutil import copyfile

# 3rd party libraries
import tensorflow as tf
from tensorflow import keras


class SaveToGDriveCallback(keras.callbacks.Callback):

    def __init__(self, model_filepath, gdrive_filepath):
        self.model_filepath = model_filepath
        self.gdrive_filepath = gdrive_filepath

    def on_epoch_end(self):
        if os.path.exists(self.gdrive_filepath):
            os.remove(self.gdrive_filepath)
            copyfile(self.model_filepath, self.gdrive_filepath)
