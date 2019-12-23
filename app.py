import os
from flask import Flask, render_template, request, send_from_directory
import tensorflow
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from helpers.config import API_KEY, CSE_KEY
from helpers.funcs import google_search, read_and_prep_images, image_dir
import models.load_model as LM

# import logging
# import urllib
# import json

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
my_model = LM.CNN()


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/search", methods=['POST'])
def search():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        image_name = upload.filename
        destination = "/".join([target, image_name])
        print("Save it to:", destination)
        upload.save(destination)
        print("The image name is :", image_name)

    from os.path import join
    img_paths = [join(image_dir, filename) for filename in
                 [image_name]]

    print(img_paths)
    test_data = read_and_prep_images(img_paths)  # for double
    print("test_data-- working", test_data.shape)
    # model loading
    # my_model = ResNet50(weights='models/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    # load_model()
    preds = my_model.predict(test_data)
    # tensorflow.keras.backend.clear_session()
    # print("Tensorflow session cleared")
    print("preds working -- working")

    for index, res in enumerate(decode_predictions(preds, top=1)[0]):
        result = res[1]
        percentage = 100 * res[2]
        result_string = '{}: {:.3f}%'.format(res[1], 100 * res[2])
        print(result_string)

    print(result)
    print(percentage)

    google_result = google_search(result, API_KEY, CSE_KEY)
    print("Our raw data is of type:", type(google_result))

    return render_template('search.html',
                           image_name_=image_name,
                           label=result,
                           accuracy=percentage,
                           result_string=result_string,
                           google_search_result=google_result
                           )

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

# tensorflow.keras.backend.clear_session()
# print("Tensorflow session cleared")

if __name__ == "__main__":
    app.run(debug = False, threaded = False, port=8080)

