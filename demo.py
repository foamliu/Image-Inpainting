# import the necessary packages
import os
import random

import cv2 as cv
import keras
import keras.backend as K
import numpy as np
from keras.preprocessing.image import (load_img, img_to_array)

from config import img_rows, img_cols, img_size
from data_generator import random_crop, separate
from model import build_model

if __name__ == '__main__':
    model_weights_path = 'models/model.00-0.0684.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    print(model.summary())

    image_folder = '/mnt/code/ImageNet-Downloader/image/resized'
    names_file = 'valid_names.txt'
    with open(names_file, 'r') as f:
        names = f.read().splitlines()

    samples = random.sample(names, 10)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        image = load_img(filename, target_size=(img_rows, img_cols))
        image = img_to_array(image)
        gt = image.copy()
        image = random_crop(image)
        x, y = separate(image)
        input = x.copy()
        x = keras.applications.resnet50.preprocess_input(x)
        x_test = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        x_test[0] = x
        out = model.predict(x_test)

        out = out[0]
        out = out * 255.0
        out = out.astype(np.uint8)

        if not os.path.exists('images'):
            os.makedirs('images')

        output = input.copy()
        output[56:168, 56:168] = out

        gt = cv.cvtColor(gt, cv.COLOR_RGB2BGR)
        input = cv.cvtColor(input, cv.COLOR_RGB2BGR)
        output = cv.cvtColor(output, cv.COLOR_RGB2BGR)

        cv.imwrite('images/{}_input.png'.format(i), input)
        cv.imwrite('images/{}_output.png'.format(i), output)
        cv.imwrite('images/{}_gt.png'.format(i), gt)

    K.clear_session()
