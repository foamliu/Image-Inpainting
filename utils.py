import multiprocessing
import os
import numpy as np
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


def get_best_model():
    import re
    pattern = 'model.(?P<epoch>\d+)-(?P<val_loss>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    files = [f for f in os.listdir('models/') if p.match(f)]
    filename = None
    epoch = None
    if len(files) > 0:
        epoches = [p.match(f).groups()[0] for f in files]
        losses = [float(p.match(f).groups()[1]) for f in files]
        best_index = int(np.argmin(losses))
        filename = os.path.join('models', files[best_index])
        epoch = int(epoches[best_index])
        print('loading best model: {}'.format(filename))
    return filename, epoch
