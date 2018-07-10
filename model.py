import keras.backend as K
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.utils import plot_model


def build_model():
    encoder = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    inputs = encoder.inputs
    outputs = encoder.outputs

    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
