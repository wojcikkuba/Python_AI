import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import layers
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.datasets import fashion_mnist

train,test = fashion_mnist.load_data()
X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]

# Rozszerzenie wymiarow
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
class_count = np.unique(y_train).shape[0]  # liczba unikatowych klas

filter_cnt = 32
units = 0.1
learning_rate = 0.0001
kernel_size = (3,3)
pooling_size = (2,2)
conv_rule = 'same'

# Budowa modelu
model = Sequential()
model.add(Conv2D(input_shape = X_train.shape[1:],filters=filter_cnt,
                 kernel_size = kernel_size, padding = conv_rule,
                 activation = 'relu'))
model.add(MaxPooling2D(pooling_size))
model.add(Conv2D(input_shape = X_train.shape[1:],filters=filter_cnt,
                 kernel_size = kernel_size, padding = conv_rule,
                 activation = 'relu'))
model.add(MaxPooling2D(pooling_size))
model.add(Flatten())
model.add(Dense(class_count,activation='relu'))
model.add(Dense(class_count,activation='softmax'))

# Kompilacja
model.compile(optimizer=Adam(learning_rate),
              loss='SparseCategoricalCrossentropy',
              metrics='accuracy')

# Trenowanie
model.fit(X_train, y_train, batch_size=32,
          epochs=class_count, validation_data=(X_test, y_test))

# Historia modelu
history = model.history.history
floss_train = history['loss']
floss_test = history['val_loss']
acc_train = history['accuracy']
acc_test = history['val_accuracy']
epochs = np.arange(0,class_count)
fig,ax = plt.subplots(1, 2, figsize=(20,10))
ax[0].plot(epochs, floss_train, label='floss_train')
ax[0].plot(epochs, floss_test, label='floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label='acc_train')
ax[1].plot(epochs, acc_test, label='acc_test')
ax[1].legend()