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
X_train_a = np.load("Xtrain2_a.npy")
print(f"tipo A é {X_train_a.shape}")

######  Convert to matrices  ######
X_train = X_train.reshape(X_train.shape[0],48,48)
y_train = y_train.reshape(y_train.shape[0],48,48)

####################################################################
### define the validation and training sets  #######################
####################################################################
percent_val = 5
n_val = int(percent_val * len(X_train) / 100)

print("Use " + str(n_val) + " images for validation")
X_val = X_train[0:n_val]
y_val = y_train[0:n_val]

X_train = X_train[n_val:]
y_train = y_train[n_val:]

#n_cut = int(len(X_train))+1 
#X_train = X_train[:n_cut]
#y_train = y_train[:n_cut]

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
    blurred = filters.gaussian(image, sigma=0.75)  #blurr to remove high frequency noise
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

# reformat the data to match what scikit-learn expects
def format_data(feature_stack, annotation):
    # feature stack  is [n_features][n_pixels]
    X = feature_stack.T     # X is [n_pixels][n_features]
    y = annotation.ravel()  # y is [n_pixels] -> 0 e 1

    return X, y

all_features = []
all_annotations = []

for index in range(len(X_train)):
    feature_stack = generate_feature_stack(X_train[index])
    if index == 0:
        # show feature images
        fig, axes = plt.subplots(1, 3, figsize=(10,10))
        # reshape(image.shape) reverts to matrix. We just need it for visualization.
        axes[0].imshow(feature_stack[0].reshape(X_train[0].shape), cmap=plt.cm.gray)
        axes[1].imshow(feature_stack[1].reshape(X_train[0].shape), cmap=plt.cm.gray)
        axes[2].imshow(feature_stack[2].reshape(X_train[0].shape), cmap=plt.cm.gray)
        plt.title("Example of features extracted")

    X , y = format_data(feature_stack , y_train[index])

    all_features.append(X)
    all_annotations.append(y)




# Concatenate the feature arrays and annotations into a single array
X_all = np.concatenate(all_features, axis=0)  # Combine all features from all images
y_all = np.concatenate(all_annotations, axis=0)  # Combine all annotations
print("Input shape", X_all.shape)
print("Annotation shape", y_all.shape)

##########  Random Tree  ################
classifier = RandomForestClassifier(max_depth=2, random_state=2)
classifier.fit(X_all, y_all)

##########  Resultados na validação   ###############
y_val_pred=[]
for image_val in X_val:
    res = classifier.predict(generate_feature_stack(image_val).T)
    res = res.reshape(X_train[0].shape)
    y_val_pred.append(res)
    if(1):
        display_2_image( image_val,res )

TP = 0
FP = 0
TN = 0
FN = 0
for im in range(len(y_val)):
    for px in range(y_val[im].shape[0]):
        for py in range(y_val[im][px].shape[0]):
            pred = y_val_pred[im][px][py]
            val  = y_val[im][px][py]
            if pred == 1 and val==1:
                TP += 1
            if pred == 1 and val==0:
                FP += 1
            if pred == 0 and val==0:
                TN += 1
            if pred == 0 and val==1:
                FN += 1
n_total_pixels = y_val.shape[0]*y_val.shape[1]*y_val.shape[2]
confusion_matrix = np.array([[100*TP/n_total_pixels,100*FN/n_total_pixels ],[ 100*FP/n_total_pixels ,100*TN/n_total_pixels ]])
print(confusion_matrix)

precision = TP/( TP + FP )
recall = TP/( TP + FN)
accuracy = (TP + TN) / (TP+FN+FP+TN)
true_neg_rate = TN/( TN + FP )
bal_acc = 0.5*(recall + true_neg_rate)
f1_score = 1/((1/recall)+(1/precision)) 

print()
print("------------------------------------------------")
print("PERFORMANCE METRICS")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f} Recall: {recall:.4f}")
print(f"F1: {f1_score:4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")



plt.show()


