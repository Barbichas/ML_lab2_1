#15.10.2024
#Machine Learning lab
#The goal is to identify craters in pictures of mars

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

random.seed(42)

X_train = np.load("Xtrain2_b.npy")
y_train = np.load("Ytrain2_b.npy")
X_test  = np.load("Xtest2_b.npy")

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



####### Function to rotate image  ###########
def rotate_image_Sofia(image):
    for i in range(int(len(image)/2)):
        ii = len(image) -1 - i

        aux = image[i]
        image[i] = image[ii]
        image[ii] = aux
    return image

####### Function for brightness variations image  ###########
def bright_image(image):
    fator_brilho = 1.5
    image_bright = np.clip(image * fator_brilho, 0, 255)#.astype(np.uint8)

    #cv2.imshow('Imagem Original', image)
    #cv2.imshow('Imagem com Mais Brilho', image_bright)

    return image_bright

####### Function for transposing the image   ###########
def transpose_image(image):
    matrix = np.array(image).reshape(48, 48)
    transpose = np.transpose(matrix)
    return np.array(transpose).reshape(48**2)

#######  Function for negative           ############
def negative_image(image):
    negative = [255-pixel for pixel in image ]
    return np.array(negative)

########   Even number of craters and plains  ##############
def equalize_crat_and_plain(X, y):
    craters = list(X[y==1] )
    plains  = list(X[y==0] )

    while(1):
        
        if len(craters) == len(plains) :
            break
        #add random plain
        aux = plains[random.randint(0, len(plains)-1)]
        aux = rotate_image_Sofia(aux)
        plains.append(aux)

        if len(craters) == len(plains) :
            break
        #add random plain 
        aux = plains[random.randint(0, len(plains)-1)]
        aux = bright_image(aux)
        plains.append(aux)

    #add labels to the data
    for i in range(len(craters)):
        craters[i] = np.concatenate( ([1] , craters[i]) )
        plains[i]  = np.concatenate( ([0] , plains[i] ) )
    X_final = craters + plains 
    random.shuffle(X_final)

    y_final = []
    for i in range( len(X_final) ):
        y_final.append( X_final[i][0] ) #put the label in y
        X_final[i]=X_final[i][1:]       #remove the label from X
    
    X_final = np.array(X_final)
    y_final = np.array(y_final)

    return X_final, y_final

def add_transpose_images(X,y):
    more_X = []
    for image in X:
        more_X.append(transpose_image(image))
    more_X = np.array(more_X)
    X_final = np.concatenate( (X,more_X) )
    y_final = np.concatenate( (y,y) )

    return X_final , y_final

def add_bright_images(X,y):
    more_X = []
    for image in X:
        more_X.append(bright_image(image))
    more_X = np.array(more_X)
    X_final = np.concatenate( (X,more_X) )
    y_final = np.concatenate( (y,y) )

    return X_final , y_final

def add_negative_images(X,y):
    more_X = []
    for image in X:
        more_X.append(bright_image(image))
    more_X = np.array(more_X)
    X_final = np.concatenate( (X,more_X) )
    y_final = np.concatenate( (y,y) )

    return X_final , y_final


#X_train, y_train = equalize_crat_and_plain(X_train,y_train)
#X_train, y_train = add_transpose_images(X_train,y_train)
#X_train, y_train = add_bright_images(X_train, y_train)
#X_train , y_train = add_negative_images(X_train, y_train)


######  looking at  images  ############
def display_image(vector, L, H):
    # Convert the vector to a 2D array (image) with shape (H, L)
    image = np.array(vector).reshape(H, L)

    # Display the image using matplotlib
    plt.imshow(image, cmap='gray')  # Using 'gray' colormap for grayscale images
    plt.axis('off')  # Hide axes

if(1):
    for i in range(20):
        fig, axs = plt.subplots(1, 2)  # Create a figure with 1 row and 2 columns of subplots
        plt.title(f"Cratter identification image {i}")
        # Display X_train[i] on the first subplot
        imageX = np.array(X_train[i]).reshape(48, 48)
        axs[0].imshow(imageX, cmap='gray')  # Using 'gray' colormap for grayscale images
        axs[0].axis('off')  # Hide axes
        # Display y_train[i] on the second subplot
        imagey = np.array(y_train[i]).reshape(48, 48)
        axs[1].imshow(imagey, cmap='gray')  # Using 'gray' colormap for grayscale images
        axs[1].axis('off')  # Hide axes
        

'''
###############################################################################
#######                Go to feature space                                #####
###############################################################################
from skimage.feature import hog
from skimage import exposure
def compute_hog(image):
    """ Compute HOG features and return them along with the visualized image. """
    features, hog_image = hog(image, 
                               orientations=6, 
                               pixels_per_cell=(48, 48), 
                               cells_per_block=(1, 1), 
                               visualize=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features, hog_image


features_train = []
if(1):
    for i_train in range(len(X_train)) :
        if(i_train % 400 == 0):
            print("Turning to features input image number " + str(i_train))
        features_train.append( compute_hog(np.array(X_train[i_train]).reshape(48, 48))[0] )
    features_train = np.array( features_train )
    np.save("SVM_features.npy",features_train)

if(0): #loading input features from file
    features_train = np.load("SVM_features.npy")


################################################################################
##########           Support Vector Machine                    #################
################################################################################


print()
print("----------------------------")
print("Support Vector Machine")
print()

svm = SVC(kernel='linear')
y_val_pred=[]
if(1): #use input space
    print("Using input space")
    svm.fit(X_train, y_train)
    y_val_pred = svm.predict(X_val)
if(0): #use feature space
    print("Using feature space")
    #train model
    svm.fit(features_train , y_train)
    #pass validation set to feature space
    features_val = []
    for i_val in range(len(X_val)) :
        if(i_val % 200 == 0):
            print("Turning to features validation image number " + str(i_val))
        features_val.append( compute_hog(np.array(X_train[i_val]).reshape(48, 48))[0] )
    features_val = np.array( features_val )
    y_val_pred = svm.predict(features_val)
if(0): #use both input and feature space
    print("Using input AND feature space, lord have mercy on this computer")
    for i in range(len(features_train)):
        i = int(i)
        features_train[i] = np.concatenate((features_train[i],X_train[i]))
    mix_train = features_train
    svm.fit(mix_train, y_train)
    #pass validation set to feature space
    features_val = []
    for i_val in range(len(X_val)) :
        if(i_val % 200 == 0):
            print("Turning to features validation image number " + str(i_val))
        features_val.append( compute_hog(np.array(X_train[i_val]).reshape(48, 48))[0] )
    features_val = np.array( features_val )
    y_val_pred = svm.predict(features_val)
    #mix output
    for i in range(len(features_val)):
        i = int(i)
        features_val[i] = np.concatenate((features_val[i],X_val[i]))
    mix_val = features_val
    y_val_pred = svm.predict(mix_val)
    


TP = 0
FN = 0
FP = 0
for i in range(len(y_val_pred)):
    if y_val[i] == 1:
        if y_val_pred[i] == 1:
            TP += 1
        else:
            FN += 1
    else:
        if y_val_pred[i] == 1:
            FP += 1
        
F1 = 2*TP/(2*TP+FN+FP)
print("Number of errors is " + str(FN+FP) +" (" +str(float(100*(FN+FP)/len(X_val))) + "%)")
print("F1 on validation is " + str(F1))

'''

plt.show()


