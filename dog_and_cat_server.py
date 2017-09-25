# main.py
from flask import Flask, current_app, request, jsonify, redirect, url_for
from PIL import Image
import requests
from io import BytesIO
import logging
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras import backend as K

#  Load saved model...
print("Loading model configuration.  One moment...")
model = load_model('./second_try.h5')
model.summary()
print("Configuration loaded.")
app = Flask(__name__)

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
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.before_first_request
def setup_logging():
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.addHandler(logging.StreamHandler())
        app.logger.setLevel(logging.INFO)

@app.route('/url/', methods=['GET'])
def predict_by_url():
    try:
        url = request.get_json()['url']
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        classes = predict(img)
        return format_response(classes)
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_response(classes):
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
Based on the trained model, the predicted class for this photo is:<br>
<br>
<h1>{0}</h1><br>
<a href="/">Click to try another image...<br>
</body>

</html>
    '''.format(class_names[round(classes.item(0))])

@app.route('/')
def root():
    return app.send_static_file('index.html')


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
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return "NO FILE SELECTED?!"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            classes = predict(file)

    return format_response(classes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
