# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.applications.vgg19 import preprocess_input

from config import img_size
from data_generator import random_crop, separate
from model import build_model

if __name__ == '__main__':
    model_weights_path = 'models/model.38-0.0171.hdf5'
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
        image = cv.imread(filename)
        image = random_crop(image)
        gt = image.copy()
        x, y = separate(image)
        input = x.copy()
        x = x[:, :, ::-1]
        x = x / 256. - 0.5
        x_test = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        x_test[0] = x
        # print('x: ' + str(x))
        # print('x_test: ' + str(x_test))

        out = model.predict(x_test)

        out = out[0]
        # print('out: ' + str(out))
        out = out * 256.0
        out = out.astype(np.uint8)

        if not os.path.exists('images'):
            os.makedirs('images')

        output = input.copy()
        output[32:96, 32:96] = out

        cv.imwrite('images/{}_input.png'.format(i), input)
        cv.imwrite('images/{}_output.png'.format(i), output)
        cv.imwrite('images/{}_gt.png'.format(i), gt)

    K.clear_session()
