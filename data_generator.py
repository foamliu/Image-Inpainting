import os
import random
from random import shuffle

import cv2 as cv
import imutils
import numpy as np
from keras.utils import Sequence

from config import batch_size, img_size, channel

image_folder = '/mnt/code/ImageNet-Downloader/image/resized'


def random_crop(image):
    orig_h, orig_w = image.shape[:2]
    u = random.randint(0, max(0, orig_w - img_size))
    v = random.randint(0, max(0, orig_h - img_size))
    result = np.zeros((img_size, img_size, 3), np.uint8)
    crop = image[v:v + img_size, u:u + img_size]
    h, w = crop.shape[:2]
    result[0:h, 0:w] = crop
    return result


def separate(image):
    x0, y0 = img_size // 4, img_size // 4
    x1, y1 = img_size * 3 // 4, img_size * 3 // 4
    img_out = image.copy()[y0:y1, x0:x1]
    img_in = image.copy()
    img_in[y0:y1, x0:x1] = 128
    return img_in, img_out


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            names_file = 'train_names.txt'
        else:
            names_file = 'valid_names.txt'

        with open(names_file, 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_size, img_size, channel), dtype=np.float32)
        batch_y = np.empty((length, img_size // 2, img_size // 2, channel), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i + i_batch]
            filename = os.path.join(image_folder, name)
            image = cv.imread(filename)
            image = random_crop(image)
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
            angle = random.choice((0, 90, 180, 270))
            image = imutils.rotate_bound(image, angle)

            x, y = separate(image)

            x = x[:, :, ::-1]
            batch_x[i_batch] = x / 256. - 0.5
            batch_y[i_batch] = y / 256.

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')


def split_data():
    names = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    num_samples = len(names)
    print('num_samples: ' + str(num_samples))

    num_train_samples = int(num_samples * 0.992)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    # with open('names.txt', 'w') as file:
    #     file.write('\n'.join(names))

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    split_data()
