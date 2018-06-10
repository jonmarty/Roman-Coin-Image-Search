# Roman Coin Image Search

This is a project which aims to apply convolutional neural networks and autoencoders to the task of identifying Ancient Roman Coins from images.

A search engine is implemented, which takes two images (one for each side of the coin) as an input, the images are concatenated. An autoencoder is used to squash the image to a lower-dimensional representation. Afterwards, the image's representation is searched against a database of images containing the fronts and backs of Roman coins. The top 50 results are returned, along with the classes associated with them.

A web app acts as a front-end to the search engine, providing an interface for inputting images and viewing query results.

## Inspiration

This project was inspired by a small difficulty that I thought would best be solved using machine-learning.

My latin teacher had ordered some ancient Roman coins from France. Me and a couple students were responsible for cleaning the coins, and, ultimately, trying to discern their identity.

I had recently been looking into convolutional neural networks. I thought this would be a great opportunity to implement my own to try and classify Roman coins.

## Setup

In order to set up the application, the images need to be downloaded and their classes determined. The setup.py script in the main directory performs this task. The steps of the setup process are listed below:

'''
python setup.py
'''

1. A web scraper looks through the results of queries on coinshome.net, collecting the links to images of coins, putting links into the data/links.txt file
2. The linked images are downloaded and stored in the Images directory
3. The classes of images are determined from their links

## Training

A model is trained using keras. To train the model, run the autoencoder.py script.

'''
python autoencoder.py
'''

## Launching Application

To launch the web app, run the web_interface.py script. This script implements a bottle application and uses the previously trained model.

'''
python web_interface.py
'''