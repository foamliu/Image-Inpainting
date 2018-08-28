import multiprocessing
import os

import cv2 as cv
import keras.backend as K
from tensorflow.python.client import device_lib


def custom_loss(y_true, y_pred):
    diff = y_pred - y_true
    return K.mean(K.sqrt(K.square(diff) + K.epsilon()))


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def get_example_numbers():
    if not os.path.isfile('train_names.txt'):
        from data_generator import split_data
        split_data()
    with open('train_names.txt', 'r') as f:
        names = f.read().splitlines()
        num_train_samples = len(names)
    with open('valid_names.txt', 'r') as f:
        names = f.read().splitlines()
        num_valid_samples = len(names)
    return num_train_samples, num_valid_samples
