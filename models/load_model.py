from tensorflow.python.keras.applications import ResNet50
from flask import Flask

app = Flask(__name__)

def CNN():
    my_model = ResNet50(weights='models/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    print("model loaded")
    return my_model


if __name__ == "__main__":
    app.run(debug=True, port=8000)