from imagenet_util import *
from utils import *
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from object_detection import save_fractal
from multiprocessing import Process
from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )

vgg_model, inception_model, resnet_model, mobilenet_model = load_model()
model = vgg_model
vgg_model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.sparse_categorical_crossentropy)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)#

train_generator=train_datagen.flow_from_directory(
    r'F:\dataset\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\train_gray',#TODO
     target_size=(300,300),#TODO
     class_mode='categorical'
)

filepath = "./weight/weights-improvement-{epoch:02d}-{loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min', save_weights_only=True)

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                                   write_images=False, update_freq='epoch', profile_batch=2,
                                   embeddings_freq=0, embeddings_metadata=None)

callbacks_list = [checkpoint, tensorboard_callback]

history=model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    callbacks=callbacks_list
)

model.save('./model.h5')