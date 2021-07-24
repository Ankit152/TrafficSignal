import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten, Dropout
from keras.layers import BatchNormalization, Dense,AveragePooling2D
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.optimizers import Adam
import os
import numpy as np
import pandas as pd
from keras.preprocessing import image


print("All libraries are imported....")

def create_model():
    model = Sequential()
    model.add(Conv2D(16,(3,3),input_shape=(32,32,3),padding="same",activation="relu",kernel_initializer="he_uniform"))
    model.add(Conv2D(16,(1,1),activation="relu",padding="same",kernel_initializer="he_uniform"))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(16,(3,3),padding="same",activation="relu",kernel_initializer="he_uniform"))
    model.add(Conv2D(16,(1,1),activation="relu",padding="same",kernel_initializer="he_uniform"))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(32,(3,3),padding="same",activation="relu",kernel_initializer="he_uniform"))
    model.add(Conv2D(32,(1,1),activation="relu",padding="same",kernel_initializer="he_uniform"))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())
    
    model.add(Dense(256,activation="relu"))
    model.add(Dense(43,"softmax"))

    return model



model = create_model()
print("Model Created....")
print(model.summary())


opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
loss = keras.losses.SparseCategoricalCrossentropy()
metrics = keras.metrics.SparseCategoricalAccuracy(name="accuracy")


model.compile(loss=loss,optimizer=opt,metrics=metrics)
print("Model compiled....")

train_dir = "./data/train"
test_dir = "./data/test"


RANDOM_SEED=123
img_size=[32,32]
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    rescale=1.0/255,
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='rgb',
    target_size=img_size,
    batch_size=64,
    class_mode='sparse',
    seed=RANDOM_SEED
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    color_mode='rgb',
    target_size=img_size,
    batch_size=8,
    class_mode='sparse',
    seed=RANDOM_SEED
)

hist = model.fit(
    train_generator,
    epochs=2,
    validation_split=0.2
)


