import numpy as np
import cv2

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

def load_model():
    # import the models for further classification experiments
    from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )

    # init the models
    vgg_model = vgg16.VGG16(weights='imagenet')
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    resnet_model = resnet50.ResNet50(weights='imagenet')
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')
    return vgg_model, inception_model, resnet_model, mobilenet_model


def fractal_preprocess(fractal):
    w = min(fractal.shape)
    fractal = fractal[0:w, 0:w]
    fractal = np.float32(fractal)
    fractal.resize((224, 224))
    fractal = cv2.cvtColor(fractal, cv2.COLOR_GRAY2RGB)
    numpy_image = img_to_array(fractal)
    image_batch = np.expand_dims(numpy_image, axis=0)
    return image_batch

