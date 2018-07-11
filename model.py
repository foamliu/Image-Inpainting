import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.layers import UpSampling2D, Conv2D, ELU, BatchNormalization
from keras.models import Model
from keras.utils import plot_model

from config import img_size, channel, kernel


def build_model():
    image_encoder = VGG19(input_shape=(img_size, img_size, channel), include_top=False, weights='imagenet',
                          pooling='None')
    # for layer in image_encoder.layers:
    #    layer.trainable = False
    inputs = image_encoder.inputs
    x = image_encoder.layers[-1].output
    print(x)

    # Decoder
    x = Conv2D(512, (kernel, kernel), padding='same', name='deconv5_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(512, (kernel, kernel), padding='same', name='deconv5_2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(512, (kernel, kernel), padding='same', name='deconv5_3', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (kernel, kernel), padding='same', name='deconv4_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(256, (kernel, kernel), padding='same', name='deconv4_2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(256, (kernel, kernel), padding='same', name='deconv4_3', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (kernel, kernel), padding='same', name='deconv3_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(128, (kernel, kernel), padding='same', name='deconv3_2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(128, (kernel, kernel), padding='same', name='deconv3_3', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (kernel, kernel), padding='same', name='deconv2_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(64, (kernel, kernel), padding='same', name='deconv2_2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(3, (1, 1), activation='sigmoid', padding='same', name='deconv1_1', kernel_initializer='he_normal')(x)

    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
