#21.10.2024
#Machine Learning lab
#The goal is to identify what pixels belong to craters in pictures of mars
#based on "https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20a_pixel_classification/scikit_learn_random_forest_pixel_classifier.html" 

import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters   #to get image features
from sklearn.ensemble import RandomForestClassifier

random.seed(42)

X_train = np.load("Xtrain2_b.npy")
y_train = np.load("Ytrain2_b.npy")
X_test  = np.load("Xtest2_b.npy")

######  Convert to matrices  ######
X_train = X_train.reshape(X_train.shape[0],48,48)
y_train = y_train.reshape(y_train.shape[0],48,48)

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

####### function to see results  #####
def display_2_image(image1, image2):
    #plt.figure()
    fig, axs = plt.subplots(1, 2)  # Create a figure with 1 row and 2 columns of subplots
    plt.title("Cratter identification image")
    axs[0].imshow(image1, cmap='gray')  # Using 'gray' colormap for grayscale images
    axs[0].axis('off')  # Hide axes
    axs[1].imshow(image2, cmap='gray')  # Using 'gray' colormap for grayscale images
    axs[1].axis('off')  # Hide axes
    return

#turn 2D image into 3 1D vectors of features
def generate_feature_stack(image):
    # determine features
    blurred = filters.gaussian(image, sigma=1.25)  #blurr to remove high frequency noise
    edges = filters.sobel(blurred)  #sobel finds edges

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

image      = X_train[0]
annotation = y_train[0]
feature_stack = generate_feature_stack(image)

# show feature images
fig, axes = plt.subplots(1, 3, figsize=(10,10))
# reshape(image.shape) reverts to matrix. We just need it for visualization.
axes[0].imshow(feature_stack[0].reshape(image.shape), cmap=plt.cm.gray)
axes[1].imshow(feature_stack[1].reshape(image.shape), cmap=plt.cm.gray)
axes[2].imshow(feature_stack[2].reshape(image.shape), cmap=plt.cm.gray)
plt.title("Example of features extracted")

# reformat the data to match what scikit-learn expects
def format_data(feature_stack, annotation):
    # feature stack  is [n_features][n_pixels]
    X = feature_stack.T     # X is [n_pixels][n_features]
    y = annotation.ravel()  # y is [n_pixels] -> 0 e 1

    return X, y

X, y = format_data(feature_stack, annotation)

print("Input shape", X.shape)
print("Annotation shape", y.shape)

##########  Random Tree  ################
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X, y)

##########  Ver resultados   ###############
for i in range(13):
    res = classifier.predict(generate_feature_stack(X_train[i]).T)
    res = res.reshape(image.shape)
    display_2_image(X_train[i],res )

plt.show()


