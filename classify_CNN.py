#23.10.2024
#Machine Learning lab
#The goal is to identify what pixels correspond to craters in pictures of mars

#############################################

import json
import math
import numpy as np
from sklearn.metrics import f1_score

##################################################

import random
import matplotlib.pyplot as plt

X_train = np.load("Xtrain2_b.npy")
y_train = np.load("Ytrain2_b.npy")
X_test  = np.load("Xtest2_b.npy")
X_train_a = np.load("Xtrain2_a.npy")
print(f"tipo A é {X_train_a.shape}")
print(f"tipo B é {X_train.shape}")


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

#n_cut = int(len(X_train)/10)+1 
#X_train = X_train[:n_cut]
#y_train = y_train[:n_cut]

#######  DESCRIPTION OF DATASET
print()
print("DESCRIPTION OF DATASET")
print()
print("Number of training images = " + str(X_train.shape[0]))
print("Number of validation images = " +str(X_val.shape[0]) + str(" ( ") + str(percent_val) + " %)")



##########  Prepare shapes for CNN  ##########
X_train = np.array(X_train).reshape(X_train.shape[0],48 , 48)
X_val   = np.array(X_val ).reshape(X_val.shape[0]  ,48 , 48)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))

y_train = y_train.reshape(y_train.shape[0] , 48 , 48, 1)
y_val   = y_val.reshape( y_val.shape[0], 48 , 48, 1)

################################################################################
###############              CNN
################################################################################

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam

# Initialize the CNN model
model = Sequential()
train_and_save = 1      ### se quiseres treinar

if (train_and_save):
    # 1st Convolutional Layer + Pooling
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsampling: (48, 48, 1) -> (24, 24, 32)

    # 2nd Convolutional Layer + Pooling
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # (24, 24, 64) -> (12, 12, 64)

    # 3rd Convolutional Layer + Pooling
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # (12, 12, 128) -> (6, 6, 128)

    # Upsampling to restore spatial dimensions
    model.add(UpSampling2D(size=(2, 2)))  # (6, 6, 128) -> (12, 12, 128)
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(UpSampling2D(size=(2, 2)))  # (12, 12, 64) -> (24, 24, 64)
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    model.add(UpSampling2D(size=(2, 2)))  # (24, 24, 32) -> (48, 48, 32)

    # Output layer: 1x1 convolution to output mask
    model.add(Conv2D(1, (1, 1), activation='sigmoid'))


    # Compile the model
    learning_rate = 0.001  # Set your desired learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print(f"Now training model. X_train shape is {X_train.shape} ,y train shape is {y_train.shape}")
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        #steps_per_epoch=len(X_train),
        epochs=8,  # Number of epochs (adjust as needed)
        validation_data=(X_val , y_val),
        #validation_steps=len(X_val)
        batch_size = 32
    )
    # Save the model
    model.save('cnn_binary_classifier.h5')

else:
    model = load_model('cnn_binary_classifier.h5')

####### testing the output shape  ####
ih = model.predict(X_val[0])
print(f"The CNN output shape is {ih.shape}")


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
##########  Resultados na validação   ###############
y_val_pred=[]
for image_val in X_val:
    res = model.predict(X_val)
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










