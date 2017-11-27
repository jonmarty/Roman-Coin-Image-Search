from os import listdir
directory_name = "Images"
image_names = listdir(directory_name)
image_names.remove('.DS_Store')
from tqdm import tqdm

import cv2
import numpy as np

imported_images = []
print("Loading Images")
#resize all images in the dataset for processing
for image_name in tqdm(image_names):
    foo = cv2.imread("Images/" + image_name)
    #foo = cv2.resize(foo, (115, 115))
    imported_images.append(foo)

#turn image into a 1D array
def process_image(image, length, width):
    out = []
    for x in range(length):
        for y in range(width):
            for z in range(3): #account for RGB
                out.append(image[x][y][z])
    return np.asarray(out)

#process all images into 1D arrays
image_set = []
#np.array([process_image(image, length = 125, width = 250) for image in imported_images])

print("Processing Images")
ix = -1
for image in tqdm(imported_images):
    ix += 1
    if image == None:
        image_names.pop(ix)
        continue
    image_set.append(process_image(image, length=125, width=250))

image_set = np.array(image_set)

print(image_set)

#Script for Autoencoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K

image_length = 125
image_width = 250
pixels = image_length * image_width * 3
hidden = 1000
corruption_level = 0.25
X = tf.placeholder("float", [None, pixels], name = 'X')
mask = tf.placeholder("float", [None, pixels], name = 'mask')

weight_max = 4 * np.sqrt(4 / (6. * (pixels + hidden)))
weight_initial = tf.random_uniform(shape = [pixels, hidden], minval = -weight_max, maxval = weight_max)

W = tf.Variable(weight_initial, name = 'W')
b = tf.Variable(tf.zeros([hidden]), name = 'b')

W_prime = tf.transpose(W)
b_prime = tf.Variable(tf.zeros([pixels]), name = 'b_prime')

def reparameterize(W = W, b = b):
    epsilon = K.random_normal(shape = (1, hidden), mean = 0.)
    return W + K.exp(b/2) * epsilon

def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X

    hidden_state = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)
    out = tf.nn.sigmoid(tf.matmul(hidden_state, W_prime) + b_prime)
    return out

out = model(X, mask, W, b, W_prime, b_prime)

cost = tf.reduce_sum(tf.pow(X-out, 2))
optimization = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

x_train, x_test = train_test_split(image_set)

sess = tf.Session()
#you need to initialize all variables
sess.run(tf.global_variables_initializer())
for i in range(1):
    for start, end in tqdm(zip(range(0, len(x_train), 128), range(128, len(x_train), 128))):
        input_ = x_train[start:end]
        mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
        sess.run(optimization, feed_dict = {X: input_, mask: mask_np})
    mask_np = np.random.binomial(1, 1 - corruption_level, x_test.shape)
    print(i, sess.run(cost, feed_dict = {X: x_test, mask: mask_np}))

masknp = np.random.binomial(1, 1 - corruption_level, image_set.shape)

#import pickle
#with open("autoencoder.pkl", 'w') as fiModel:
#    pickle.dump(sess, fiModel)

tSav = tf.train.Saver()
tSav.save(sess, "autoencoder.ckpt")

######SPLIT#####


def hidden_state(X, mask, W, b):
    tilde_X = mask * X

    state = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)
    return state


def euclidean_distance(arr1, arr2):
    x = 0
    for i in range(len(arr1)):
        x += pow(arr1[i] - arr2[i], 2)
    return np.sqrt(x)


def search(image):
    hidden_states = [sess.run(hidden_state(X, mask, W, b),
                              feed_dict={X: im.reshape(1, pixels),
                                         mask: np.random.binomial(1, 1 - corruption_level, (1, pixels))})
                     for im in image_set]
    query = sess.run(hidden_state(X, mask, W, b),
                     feed_dict={X: image.reshape(1, pixels),
                                mask: np.random.binomial(1, 1 - corruption_level, (1, pixels))})
    starting_state = int(np.random.random() * len(hidden_states))  # choose random starting state
    best_states = [imported_images[starting_state]]
    distance = euclidean_distance(query[0],
                                  hidden_states[starting_state][0])  # Calculate similarity between hidden states
    for i in range(len(hidden_states)):
        dist = euclidean_distance(query[0], hidden_states[i][0])
        if dist <= distance:
            distance = dist  # as the method progresses, it gets better at identifying similiar images
            best_states.append(imported_images[i])
    if len(best_states) > 0:
        return best_states
    else:
        return best_states[len(best_states) - 50:]
"""
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

image_name = "Images/1_AE_-Bronze-Roman_Empire_(27BC_395)-250-125-q3DBwcI0HCwAAAEmc6Y6TOhH.jpg" #Image to be used as query
image = cv2.imread(image_name)
#image = cv2.resize(image, (115, 115))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis('off')
image = process_image(image, length=125, width=250)

results = search(image) #Search the database using image

print(len(results))
slots = 0
plt.figure(figsize = (125,125))
for im in results[::-1]: #reads through results backwards (more similiar images first)
    plt.subplot(10, 10, slots+1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    slots += 1
"""
