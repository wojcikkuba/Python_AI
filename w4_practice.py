from keras.utils.vis_utils import plot_model
#from tensorflow.keras.layers import concatenate
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, concatenate
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import pandas as pd
import tensorflow as tf

data = mnist.load_data()

X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values
kernel_size = (3, 3)
poling_size = (3, 3)
act_func = 'relu'
padding = 'same'


def add_inseption_module(input_tensor):
    act_func = "relu"
    paths = [
        [
            Conv2D(256, (3, 3), padding='same', activation=act_func),
            MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
            Conv2D(128, (3, 3), padding='same', activation=act_func),
            Conv2D(64, (3, 3), padding='same', activation=act_func),
            MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
            Conv2D(64, (3, 3), padding='same', activation=act_func)
        ],
        [
            Conv2D(128, (3, 3), padding='same', activation=act_func),
            MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
            Conv2D(64, (3, 3), padding='same', activation=act_func),
            MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
            Conv2D(64, (3, 3), padding='same', activation=act_func),
            Conv2D(64, (3, 3), padding='same', activation=act_func)
        ],
        [
            Conv2D(256, (3, 3), padding='same', activation=act_func),
            Conv2D(128, (3, 3), padding='same', activation=act_func),
            Conv2D(128, (3, 3), padding='same', activation=act_func),
            MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
            Conv2D(64, (3, 3), padding='same', activation=act_func),
            Conv2D(64, (3, 3), padding='same', activation=act_func)
        ],
        [
            Conv2D(128, (3, 3), padding='same', activation=act_func),
            Conv2D(64, (3, 3), padding='same', activation=act_func),
            MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
            Conv2D(32, (3, 3), padding='same', activation=act_func)
        ]
    ]
    for_concat = []
    for path in paths:
        output_tensor = input_tensor
        for layer in path:
            output_tensor = layer(output_tensor)
        for_concat.append(output_tensor)
    return concatenate(for_concat)


output_tensor = input_tensor = Input(X_train.shape[1:])
output_tensor = tf.keras.layers.BatchNormalization()(output_tensor)
output_tensor = add_inseption_module(output_tensor)
output_tensor = Flatten()(output_tensor)
output_tensor = Dense(512, activation=act_func)(output_tensor)
output_tensor = Dense(256, activation=act_func)(output_tensor)
output_tensor = Dense(128, activation=act_func)(output_tensor)
output_tensor = Dense(10, activation='softmax')(output_tensor)

ANN = Model(inputs=input_tensor, outputs=output_tensor)
ANN.compile(loss='categorical_crossentropy',
            metrics='accuracy', optimizer='adam')


plot_model(ANN, show_shapes=True, to_file='graph.png')