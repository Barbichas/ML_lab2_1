#21.10.2024
#Machine Learning lab
#The goal is to identify what pixels belong to craters in pictures of mars

import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters   #to get image features

random.seed(42)

X_train = np.load("Xtrain2_b.npy")
y_train = np.load("Ytrain2_b.npy")
X_test  = np.load("Xtest2_b.npy")

######  Convert to matrices  ######
for i in range(len(X_train)):
    X_train[i] = X_train[i].reshape(48,48)
    y_train[i] = y_train[i].reshape(48,48)

####################################################################
### define the validation and training sets  #######################
####################################################################
percent_val = 30
n_val = int(percent_val * len(X_train) / 100)

print("Use " + str(n_val) + " images for validation")
X_val = X_train[0:n_val]
y_val = y_train[0:n_val]

X_train = X_train[n_val:]
y_train = y_train[n_val:]



#######  DESCRIPTION OF DATASET

print()
print("DESCRIPTION OF DATASET")
print()
print("Number of training images = " + str(X_train.shape[0]))
print("Number of validation images = " +str(X_val.shape[0]) + str(" ( ") + str(percent_val) + " %)")


def generate_feature_stack(image):
    # determine features
    blurred = filters.gaussian(image, sigma=2)
    edges = filters.sobel(blurred)

    # collect features in a stack
    # The ravel() function turns a nD image into a 1-D image.
    # We need to use it because scikit-learn expects values in a 1-D format here. 
    feature_stack = [
        image.ravel(),
        blurred.ravel(),
        edges.ravel()
    ]
    
    # return stack as numpy-array
    return np.asarray(feature_stack)

image = X_train[0]
feature_stack = generate_feature_stack(image)

# show feature images
fig, axes = plt.subplots(1, 3, figsize=(10,10))

# reshape(image.shape) is the opposite of ravel() here. We just need it for visualization.
axes[0].imshow(feature_stack[0].reshape(image.shape), cmap=plt.cm.gray)
axes[1].imshow(feature_stack[1].reshape(image.shape), cmap=plt.cm.gray)
axes[2].imshow(feature_stack[2].reshape(image.shape), cmap=plt.cm.gray)


