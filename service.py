from flask import Flask, request, jsonify
from waitress import serve
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model
import cv2
import os
from PIL import Image

app = Flask(__name__)


labels=['normal','pneumonia']
input_shape = (224, 224, 3)
img_input = Input(shape=input_shape)
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=img_input, input_shape=input_shape)
x = base_model.output
print(base_model.output_shape[1:])
x = Flatten(input_shape=base_model.output_shape[1:])(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid", name="predictions")(x)
model = Model(inputs=img_input, outputs=predictions)
model.load_weights('vgg16_rgb_binary_xray_class_equal_weights.best.hdf5')


def prepare(array):
    img = Image.fromarray(array)
    new_array = cv2.resize(img, (224, 224))
    return new_array.reshape(-1, 224, 224, 3)


@app.route("/1", methods=['GET', 'POST'])
def check():
    if request.method == 'POST':
        height = int(request.form['height'])
        width = int(request.form['width'])
        image = request.form.getlist('image')
        image = [int(i) for i in image]
        image = np.array(image, dtype='uint8')
        print(image)
        print(type(image[1]))
        image2d = image.reshape(height, width)
        image2d = cv2.resize(image2d, (224, 224))
        print('image=', image, 'height=', height, 'width=', width)
        prediction = model.predict([prepare(image2d.reshape(-1, 224, 224, 3))])
        return jsonifyP{'diagnosis':labels[int(prediction[0])],'confidence':prediction}


if __name__ == "__main__ ":
    serve(app, listen='*:5000')
