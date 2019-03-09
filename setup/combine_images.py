import numpy as np
import os
import cv2

# Create a blank array to store images in
images = []

# Iterate through the "images" directory
for img in os.listdir("images"):
    
    # Load image in grayscale
    img = cv2.imread("images/%s" % img, 0)
    img = cv2.resize(img, (128,128))
    images.append(img)

# Stack images together into one numpy array
data = np.stack(images, axis=0)

# Add a new dimension to the array to fit the input shape for the model
data = np.expand_dims(data, axis=-1)

# Save numpy array to file
np.save("data/images", data)
