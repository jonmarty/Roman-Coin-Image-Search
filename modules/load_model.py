from keras.models import model_from_json
from PIL import Image
import numpy as np

def load_model(model_name):
    loaded_model = model_from_json(open(model_name+".json", "r").read())
    loaded_model.load_weights(model_name+".h5")
    return loaded_model
def load_image( infilename, resize = None) :
    img = Image.open(infilename)
    img.load()
    img = img.convert("RGB")
    if resize == "big":
        img = img.resize((125,250))
    elif resize == "small":
        img = img.resize((125,125))
    data = np.asarray(img, dtype="float32")
    return data
def has_white_border(image, size = 1):
    for f in range(3):
        for i in range(image.shape[0]):
            for h in range(size):
                if image[h][0][f] != 255.:
                    return False
                if image[i][-1-h][f] != 255.:
                    return False
        for i in range(image.shape[1]):
            for h in range(size):
                if image[h][i][f] != 255.:
                    return False
                if image[-1-h][i][f] != 255.:
                    return False
    return True