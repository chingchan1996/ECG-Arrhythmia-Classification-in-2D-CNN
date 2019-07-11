from keras.layers import Dense, Dropout, Conv2D, Input, MaxPool2D, Flatten
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import ELU



def proposed_model(nb_classes=8, input_h=128, input_w=128):
    input_shape = (input_h, input_w, 3)

    inputs = Input(input_shape)

    # layer 1
    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(inputs)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # layer 2
    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # layer3
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer4
    x = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # layer5
    x = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # layer6
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer7
    x = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # layer 8
    x = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')(x)
    x = ELU()(x)
    x = BatchNormalization()(x)

    # layer 9
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)

    # layer 10
    x = Dense(2048)(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs, x)
    return model