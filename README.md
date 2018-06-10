# Roman Coin Image Search

This is a project which aims to apply convolutional neural networks and autoencoders to the task of identifying Ancient Roman Coins from images.

A search engine is implemented, which takes two images (one for each side of the coin) as an input, the images are concatenated. An autoencoder is used to squash the image to a lower-dimensional representation. Afterwards, the image's representation is searched against a database of images containing the fronts and backs of Roman coins. The top 50 results are returned, along with the classes associated with them.