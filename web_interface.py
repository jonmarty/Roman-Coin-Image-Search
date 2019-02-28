import bottle
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Model
import sys
sys.path.append("./modules")
from load_model import load_model, load_image, has_white_border
from tqdm import tqdm
import os
import argparse

# Determine the keyword to load the encoder
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--keyword", type=str)
args = parser.parse_args()
keyword = args.keyword

encoder = load_model("models/encoder.%s" % keyword)
encoder.compile(optimizer = "rmsprop", loss = "binary_crossentropy")

filenames = ["images/" + o for o in os.listdir("images") if o.endswith(".jpg") and (not o.startswith("."))]

X = np.asarray([load_image(o) for o in tqdm(filenames)])
X = X.astype('float32') / 255.
X = X.reshape((len(X), np.prod(X.shape[1:])))

X_encoded = [encoder.predict(np.array([x])) for x in tqdm(X)]

#############################

def euclidean_distance(arr1, arr2):
    x = 0
    for i in range(len(arr1)):
        x += pow(arr1[i] - arr2[i], 2)
    return np.sqrt(x)

def search(image, result_size = 50):
    encoded_image = encoder.predict(image)
    distances = np.array([np.linalg.norm(encoded_image-image) for image in X_encoded], dtype="float32")
    top50 = distances.argsort()[-result_size:][::-1]
    return [filenames[i] for i in top50]

@bottle.get("/")
def get_main_page():
    return open("pages/Main.html", "r").read()

@bottle.post("/")
def perform_search():
    os.system("del image1.jpg")
    os.system("del image2.jpg")
    bottle.request.files.get("image1").save("image1.jpg")
    bottle.request.files.get("image2").save("image2.jpg")
    bottle.redirect("/search")

@bottle.get("/search")
def bottle_search():
    image1 = load_image("image1.jpg", resize = "small")
    image2 = load_image("image2.jpg", resize = "small")
    image = np.concatenate((image1, image2), axis=1)
    image = image.reshape((1, np.prod(image.shape)))
    results = search(image, result_size=50)
    classes = [" ".join(i.split("-")[0:3]).replace("_"," ").replace("(* ", "BC to ").replace("Images/","").strip() for i in results]
    blocks = [{"image":r,"label":c} for r, c in zip(results, classes)]
    blocks = "".join(["<div class=\"gallery\"><img src=\"{}\" width=\"250\" height=\"125\"><div class=\"desc\">{}</div></div>".format(i["image"].replace("Images/",""),i["label"]) for i in blocks])
    return open("pages/Search.html","r").read().format(blocks)

#@bottle.get("/static/img/<filepath:re:.*\.(jpg|png|gif|ico|svg)>")
#def img(filepath):
#    return bottle.static_file(filepath, root="static/img")

#@bottle.route('/static/<filename>')
#def server_static(filename):
#    return bottle.static_file(filename, root='/path/to/your/static/files')

@bottle.get("/<filename>")
def image_load(filename):
    return bottle.static_file(filename, root="Images")

@bottle.get("/style/<filename>")
def css_load(filename):
    return bottle.static_file(filename, root="style")

if __name__ == "__main__":
    bottle.run(host = "localhost", port = 1234)