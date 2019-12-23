import numpy as np
from googleapiclient.discovery import build
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 224
image_dir = 'images'

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    search_result = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return search_result

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    # img_array = np.expand_dims(img_array, axis=0)
    output = preprocess_input(img_array)
    return (output)