#dependencies (numpy, matplotlib, and keras)
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
import cv2
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from LoadModel import load_image

#hyperparameters
batch_size = 1
original_dim = 93750
latent_dim = 20
intermediate_dim = 256
nb_epoch = 5
epsilon_std = 1.0

#encoder
x = Input(batch_shape = (batch_size, original_dim))
h = Dense(intermediate_dim, activation = 'relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim, activation = "softmax")(h)

print(z_mean)
print(z_log_var)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape = (batch_size, latent_dim), mean = 0.)
    return z_mean + K.exp(z_log_var/2) * epsilon

#note that "output_shape" isn't necessarily with the Tensorflow backend
z = Lambda (sampling, output_shape = (latent_dim,))([z_mean, z_log_var])

#latent hidden state

print(z)

#decoder
decoder_h = Dense(intermediate_dim, activation = 'relu')
decoder_mean = Dense(original_dim, activation = 'sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

print(x_decoded_mean)

#TODO: Make loss function stop returning nan
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss) #loss="binary_crossentropy")

filenames = ["Images/" + o for o in os.listdir("Images") if o.endswith(".jpg") and (not o.startswith("."))]

#x_train = np.array([np.asarray(cv2.imread(o)) for o in tqdm(filenames)])
#print(x_train[0])
X = np.asarray([load_image(o) for o in tqdm(filenames)])

X = X.astype('float32') / 255.
X = X.reshape((len(X), np.prod(X.shape[1:])))

x_train, x_test = train_test_split(X, test_size = 0.5)

vae.fit(x_train, x_train, shuffle = True, epochs = nb_epoch, batch_size = batch_size, validation_data = (x_test, x_test), verbose = 1)

encoder = Model(x, z_mean)

encoder_json = encoder.to_json()
json_file = open("encoder2.json", "w")
json_file.write(encoder_json)
json_file.close()
encoder.save_weights("encoder2.h5")