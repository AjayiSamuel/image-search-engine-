import os
import urllib
import json
from flask import Flask, render_template, request, send_from_directory, jsonify
import numpy as np
import tensorflow
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


image_size = 224
image_dir = '/home/mlg/Documents/Projects/image search engine/images'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def main():
    return render_template('upload.html')

@app.route("/upload", methods=['POST'])
def upload():
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
    # img_path = [join(image_dir, image_name)]
    img_paths = [join(image_dir, filename) for filename in
                 [image_name]]

    print(img_paths)

    def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
        imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
        img_array = np.array([img_to_array(img) for img in imgs])
        output = preprocess_input(img_array)
        return (output)

    test_data = read_and_prep_images(img_paths)  # for double
    print("test_data-- working", test_data.shape)
    my_model = ResNet50(weights='/home/mlg/Documents/Projects/image search engine/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    preds = my_model.predict(test_data)
    tensorflow.keras.backend.clear_session()
    print("preds working -- working")

    for index, res in enumerate(decode_predictions(preds, top=1)[0]):
        result = res[1]
        percentage = 100 * res[2]
        result_string = '{}: {:.3f}%'.format(res[1], 100 * res[2])
        print(result_string)

    print(result)
    print(percentage)

    from googleapiclient.discovery import build
    my_api_key = "AIzaSyACl5n-r256y44cZjFnPcbrrN4zMGMNpBM"
    my_cse_id = "007730378425504031301:bytbtow3f9u"

    def google_search(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        search_result = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return search_result

    google_result = google_search(result, my_api_key, my_cse_id)
    print("Our raw data is of type:", type(google_result))
    # print(google_result)
    # json_google_result = json.dumps(google_result)
    # print(json_google_result)
    # print("Our new data is of type:", type(json_google_result))


    return render_template('search.html', image_name_=image_name, label=result, accuracy=percentage, result_string_= result_string, google_search_result = google_result)


@app.route("/predict", methods=['POST'])
def predict():
    return True


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == "__main__":
    app.run(debug=True, port = 8000)
