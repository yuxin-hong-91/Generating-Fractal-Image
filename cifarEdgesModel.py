#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    10-Nov-2022 22:36:34

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential

def create_model(lw=False, weights='./weights/lastest/weights-improvement-08-2.19.h5'):
    input_unnormalized = keras.Input(shape=(32,32,1), name="input_unnormalized")
    input = SubtractConstantLayer((32,32,1), name="input_")(input_unnormalized)
    conv_1_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(input)
    conv_1 = layers.Conv2D(32, (4,4), name="conv_1_")(conv_1_prepadded)
    batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(conv_1)
    layer_1 = layers.Activation('swish')(batchnorm_1)
    maxpool_1 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_1)
    conv_2_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(maxpool_1)
    conv_2 = layers.Conv2D(64, (4,4), name="conv_2_")(conv_2_prepadded)
    batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(conv_2)
    layer_2 = layers.Activation('swish')(batchnorm_2)
    maxpool_2 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(layer_2)
    conv_3_prepadded = layers.ZeroPadding2D(padding=((1,1),(1,1)))(maxpool_2)
    conv_3 = layers.Conv2D(256, (4,4), name="conv_3_")(conv_3_prepadded)
    batchnorm_3 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_3_")(conv_3)
    layer_3 = layers.Activation('swish')(batchnorm_3)
    pool = layers.GlobalAveragePooling2D(keepdims=True)(layer_3)
    dropout = layers.Dropout(0.400000)(pool)
    fc = layers.Reshape((1, 1, -1), name="fc_preFlatten1")(dropout)
    fc = layers.Dense(101, name="fc_")(fc)
    batchnorm_4 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_4_")(fc)
    softmax = layers.Softmax()(batchnorm_4)
    classoutput = layers.Flatten()(softmax)

    model = keras.Model(inputs=[input_unnormalized], outputs=[classoutput])
    if lw:
        model.load_weights(weights)
    return model

## Helper layers:

class SubtractConstantLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(SubtractConstantLayer, self).__init__(name=name)
        self.const = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        return input - self.const

def simple_cnn(lw=False, weights='./weights/simplecnn_edter/weights-improvement-77-2.70.h5'):
    model = Sequential([
        layers.Conv2D(16, (4, 4), input_shape=(32, 32, 1), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (4, 4), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (4, 4), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAvgPool2D(),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(100, activation='softmax')
    ])
    if lw:
        model.load_weights(weights)
    return model