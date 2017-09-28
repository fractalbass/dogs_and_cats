from flask import Flask, current_app, request, jsonify, redirect, url_for,  send_from_directory
from PIL import Image
import logging
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras import backend as K

#  Load saved model...
print("Loading model configuration.  One moment...")
model = load_model('./second_try.h5')
model.summary()
print("Configuration loaded.")

#  This is by convention based on how the model was trained...
class_names = {0: 'CAT', 1: 'DOG'}

# Configure image specifications
img_width, img_height = 150, 150
K.set_image_dim_ordering('tf')
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# We save the image and then process it.
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.before_first_request
def setup_logging():
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.addHandler(logging.StreamHandler())
        app.logger.setLevel(logging.INFO)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_response(classes, filename):
    v = {"class": class_names[round(classes.item(0))]}
    print("V is {0}".format(v))
    j = json.dumps(v)
    print("J is {0}".format(j))
    return '''
    <!DOCTYPE html>
<html>
<head>
    <title>Keras/Flask Image Processor Example</title>
</head>
<body>
<h1>
    Deep Learning Processing Server Example
</h1>
<img width="180" src="/images/{0}"></img><br>
Based on the trained model, the predicted class for this photo is:<br>
<br>
<h1>{1}</h1><br>
<a href="/index.html">Click to try another image...</a><br>
<br><br>
<hr>
More about this application can be found 
<a href="http://datascience.netlify.com/general/2017/09/25/data_science_21.html">http://datascience.netlify.com</a>
<br><br>
Questions or comments about this app?<br>
Please contact me at mporter@paintedharmony.com<br>
<br>
Code for this application is <a href="https://github.com/fractalbass/dogs_and_cats">available in this GitHub repo.</a>
<br><br>
</body>

</html>
    '''.format(filename, j)

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('uploads', path)


def predict(f):
    p = "{0}/{1}".format(UPLOAD_FOLDER, f.filename)
    img = Image.open(p)

    if img.size != (img_width, img_height):
        img = img.resize((img_width, img_height))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    classes = model.predict(x)
    logging.info("Classes found: {0}".format(classes))
    return classes

@app.route('/file/', methods=['GET', 'POST'])
def predict_by_file():
    classes = "ERROR PROCESSING FILE!"
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            return "File no found."

        file = request.files['file']

        # If there is not file, send a message
        if file.filename == '':
            return "No file appears to be selected."

        # If it is an invalid file, send a message.
        if allowed_file(file.filename)==False:
            return "File type not allowed."

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            classes = predict(file)

    return format_response(classes, file.filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
