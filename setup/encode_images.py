import sys
sys.path.append("./modules")
import argparse
import numpy as np
import keras
from load_model import load_model, load_image, has_white_border

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--keyword", type=str)
args = parser.parse_args()

encoder = load_model("models/encoder.%s" % args.keyword)
encoder.compile(optimizer = "rmsprop", loss = "binary_crossentropy")

X = np.load("data/images.npy")
X_encoded = np.array(encoder.predict(X))[0,:,:]
np.save("data/encoded.%s" % args.keyword, X_encoded)
