import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras import applications  

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
model = load_model('my_cifar10_model.h5')
global graph
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    path = os.path.join('uploads', filename)
    #Step 2
    my_image_re = resize(my_image, (32,32,3))
    
    #Step 3
    with graph.as_default():
      set_session(sess)
      #probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
      #print(probabilities)
      #Step 4

      probabilities = test_single_image(path)
      (d1, p1) = probabilities[0]
      (d2, p2) = probabilities[1]
     
      predictions = {

        "class1": d1,
        "class2": d2,
        
         "prob1": p1,
         "prob2": p2,
         
      }

#Step 5
    return render_template('predict.html', predictions=predictions)
    
def read_image(file_path):
    print("[INFO] loading and preprocessing image...")  
    image = load_img(file_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image
    
def test_single_image(path):
    disease = ['NORMAL', 'COVID-19-PNEUMONIA']
    images = read_image(path)
  
    model = load_model('fill in the model path')
    preds = model.predict(images)
    print("**********************************",preds)
    print("**********************************",disease)
    probabilities = []
    probabilities.append((disease[0],preds[0][0]))
    probabilities.append((disease[1],preds[0][1]))
    return probabilities




app.run(host='0.0.0.0', port=80)