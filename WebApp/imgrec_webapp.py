import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras import applications
import os
import glob
import h5py
import shutil
#import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
#import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout , GlobalAveragePooling2D
#import cv2
import tensorflow as tf
from keras import backend as K
color = sns.color_palette()
K.common.set_image_dim_ordering('th')






app = Flask(__name__)

from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
print("Loading model")
global sess
sess = tf.Session()
set_session(sess)
global model
#model = load_model('my_cifar10_model.h5')
global graph
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename, force="auto"))
    return render_template('index.html')

@app.route('/prediction/<filename>/<force>')
def prediction(filename, force):
    plt.imread(os.path.join('uploads', filename))
    path = os.path.join('uploads', filename)
    with graph.as_default():
      set_session(sess)
      probabilities, ensemble_status = test_single_image(path,force)
      ensembleResult = {"status": ensemble_status}
      (d1, p1) = probabilities[0]
      (d2, p2) = probabilities[1]
     
      predictions = {
        "class1": d1,
        "class2": d2,
         "prob1": round(p1*100, 2),
         "prob2": round(p2*100, 2),
      }
    print(force)

    return render_template('predict.html', predictions=predictions, ensembleResult=ensembleResult)
    
def read_image(file_path):
    print("[INFO] loading and preprocessing image...")  
    image = load_img(file_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image
    
def test_single_image(path,force):
    disease = ['NORMAL', 'PNEUMONIA']
    image = read_image(path)
    #vgg16 = applications.VGG16(include_top=False, weights='imagenet') 
    #bt_prediction = vgg16.predict(images)
    ensemble_status = "Inception"
    preds = inception(image)
    probabilities = []
    if force == "force":
        probabilities = ensemble(image, preds)
        ensemble_status = "Ensemble of models"
        return probabilities, ensemble_status

    elif force == "vgg":
        preds = vgg(image)
        probabilities.append((disease[0], preds[0][0]))
        probabilities.append((disease[1], preds[0][1]))
        ensemble_status = "VGG16"
        return probabilities, ensemble_status
    elif force == "inception":
        probabilities.append((disease[0], preds[0][0]))
        probabilities.append((disease[1], preds[0][1]))
        ensemble_status = "Inception"
        return probabilities, ensemble_status

    elif force == "resnet":
        preds = resNet(image)
        probabilities.append((disease[0], preds[0][0]))
        probabilities.append((disease[1], preds[0][1]))
        ensemble_status = "ResNet"
        return probabilities, ensemble_status
    else:
        if preds[0][0] < 0.67 or preds[0][1] < 0.67:
           probabilities.append((disease[0],preds[0][0]))
           probabilities.append((disease[1],preds[0][1]))
        else:
           probabilities = ensemble(image,preds)
           ensemble_status = "Ensemble of models"
    return probabilities, ensemble_status

def vgg(image):
    vgg = load_model('VGG.h5')
    return vgg.predict(image)

def ensemble(image, PI):
    disease = ['NORMAL', 'PNEUMONIA']
    probabilities = []
    PV = vgg(image)
    PR = resNet(image)
    
    preds = ( np.array(PI) + np.array(PV) +  np.array(PR)) / 3.0
    probabilities.append((disease[0],preds[0][0]))
    probabilities.append((disease[1],preds[0][1]))
    
    return probabilities

def inception(image):
    from keras.applications.inception_v3 import InceptionV3
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(2, activation='sigmoid')(x)

    base_model.load_weights("inception_v3_weights.h5")
    model = Model(inputs=base_model.input, outputs=predictions)
    opt = RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.load_weights("Inception.h5")
    return model.predict(image)

def resNet(image):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    base_model = ResNet50(input_shape=(224, 224, 3),
                          weights='imagenet',
                          include_top=False,
                          pooling='avg')
    feature_inputs = Input(shape=base_model.output_shape, name='top_model_input')
    x = Dense(50, activation='relu', name='fc1')(feature_inputs)
    x = BatchNormalization()(x)
    outputs = Dense(2, activation='softmax', name='fc2')(x)
    top_model = Model(feature_inputs, outputs, name='top_model')
    inputs = Input(shape=(224, 224, 3))
    model = get_fine_tuning_model(base_model, top_model, inputs, "transfer_learning")
    opt = RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    model.load_weights("ResNet.h5")
    return model.predict(image)

def get_fine_tuning_model(base_model, top_model, inputs, learning_type):
    if learning_type == 'transfer_learning':
        print("Doing transfer learning")
        K.set_learning_phase(0)
        base_model.trainable = False
        features = base_model(inputs)
        outputs = top_model(features)
    else:
        print("Doing fine-tuning")
        base_model.trainable = True
        features = base_model(inputs)
        outputs = top_model(features)
    return Model(inputs, outputs)




app.run(host='0.0.0.0', port=80)