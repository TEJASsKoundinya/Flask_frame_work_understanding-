

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_inception.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Disease is bell Bacterial spot and caused by four species of Xanthomonas-fungicide "
    elif preds==1:
        preds="The Disease is bell early blight and two different related fungi causes it. "
    elif preds==2:
        preds="The Disease is late blight fungus-like organisms also called water molds"
    elif preds==3:
        preds="The Disease is leaf mold caused by fungus Passalora fulva"
    elif preds==4:
        preds="The Disease is Late blight fungus-like organisms also called water molds, but they are not true fungi "
    elif preds==5:
        preds="The Disease is spider mites two spotted-pesticides to mites called miticide "
    elif preds==6:
        preds="The Disease is Yellow Leaf Curl Virus found in tomato plant virus transmitted by whitefly Bemisia "
    elif preds==7:
        preds="The Disease is because of leaf curl virus "
    elif preds==8:
        preds="The Disease is because of mosaic virus. The causal viruses are spread by aphids and other insects, mites, fungi"
    elif preds==9:
        preds="The Disease is Bacterial spot caused by four species of Xanthomonas-fungicide "
    elif preds==10:
        preds="The Disease is Bacterial  "
    elif preds==11:
        preds="The Disease is Bacterial "
    elif preds==12:
        preds="The Disease is Bacterial "
    elif preds==13:
        preds="The Disease is Bacterial "
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
