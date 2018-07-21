import keras.backend as K
from keras.layers import Input, ZeroPadding2D, Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Reshape, \
    Concatenate, Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import plot_model

from config import channel, kernel
from config import img_rows, img_cols
from custom_layers.unpooling_layer import Unpooling


def build_model():
    # Encoder
    img_input = Input(shape=(img_rows, img_cols, channel))
    x = ZeroPadding2D((1, 1))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv1_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv1_2')(x)
    orig_1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv2_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv2_2')(x)
    orig_2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv3_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv3_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv3_3')(x)
    orig_3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv4_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv4_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv4_3')(x)
    orig_4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv5_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv5_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv5_3')(x)
    orig_5 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Add Fully Connected Layer
    x_fc = Flatten()(x)
    x_fc = Dense(4096, activation='relu')(x_fc)
    x_fc = Dropout(0.5)(x_fc)
    x_fc = Dense(4096, activation='relu')(x_fc)
    x_fc = Dropout(0.5)(x_fc)
    x_fc = Dense(1000, activation='softmax')(x_fc)
    model = Model(img_input, x_fc)

    # Loads ImageNet pre-trained data
    weights_path = 'models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_5)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_5)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='deconv5_1',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='deconv5_2',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='deconv5_3',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_4)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_4)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='deconv4_1',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='deconv4_2',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='deconv4_3',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_3)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_3)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='deconv3_1',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='deconv3_2',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv3_3',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_2)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_2)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv2_1',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='deconv2_2',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    the_shape = K.int_shape(orig_1)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_1)
    xReshaped = Reshape(shape)(x)
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    x = Unpooling()(together)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='deconv1_1',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='deconv1_2',
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)

    x = Conv2D(3, (1, 1), padding='same', name='pred', kernel_initializer='he_uniform')(x)

    outputs = x
    model = Model(inputs=img_input, outputs=outputs)
    return model


if __name__ == '__main__':
    model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
