import os
import csv
import cv2
import numpy as np
import sklearn
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten,Dropout,Lambda
from keras.layers.convolutional import Cropping2D,Convolution2D
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization

#Data Augmentation or manipulation functions in this cell

def imgRead(img_path):
    #Read image is in BGR format
    img = cv2.imread(img_path)
    #Convert to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
def Imflip(img):
    return cv2.flip(img,1)

#Not used, incorporate cropping in the Model arch so that its common to train,valid and test sets
def ImCrop(img):
    #Actual image dim is 160x320 and removing 50 from top and 20 from bottom
    return img[50:140,:,:]

def throwaway_func(samples):
    num_samples = len(samples)
    angles = []
    new_samples = []
    for sample in samples:
        center_angle = float(sample[3])
        angles.append(center_angle)
    plt.hist(angles)
    plt.show()
    angles = []
    #eliminate some samples with steering angle of close to 0
    for sample in samples:
        center_angle = float(sample[3])
        if center_angle >= 0.0 and center_angle <= 0.20 and np.random.random_sample() >= 0.5:
            new_samples.append(sample)
            angles.append(center_angle)
        elif center_angle < 0.0 or center_angle > 0.20:
            new_samples.append(sample)
            angles.append(center_angle)
    plt.hist(angles)
    plt.show()
    return new_samples   
        

def t_generator(samples, batch_size=32):
    num_samples = len(samples)
    src_path = 'more_d/IMG/'
    #Delta defined for steering angles from left and right cameras 
    st_delta = 0.25
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                c_im_name = src_path + batch_sample[0].split('/')[-1]
                l_im_name = src_path + batch_sample[1].split('/')[-1]
                r_im_name = src_path + batch_sample[2].split('/')[-1]
                
                #Read the images
                c_image = imgRead(c_im_name)
                l_image = imgRead(l_im_name)
                r_image = imgRead(r_im_name)
                #Flip the center image to generate a flipped image
                fl_image = Imflip(c_image)
                
                #Read the corresp angle
                center_angle = float(batch_sample[3])
                
         
                #Append the center, left, right and flipped images and their st angles
                images.append(c_image)
                angles.append(center_angle)
                
                images.append(l_image)
                angles.append(center_angle+st_delta)
                
                images.append(r_image)
                angles.append(center_angle-st_delta)
                
                images.append(fl_image)
                angles.append(-center_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def v_generator(samples, batch_size=32):
    num_samples = len(samples)
    src_path = 'more_d/IMG/'
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                c_im_name = src_path + batch_sample[0].split('/')[-1]

                #Read the images
                c_image = imgRead(c_im_name)
               
                #Read the corresp angle
                center_angle = float(batch_sample[3])
                
                #Append the center validation sample and its steering angle.
                images.append(c_image)
                angles.append(center_angle)


            X_valid = np.array(images)
            y_valid = np.array(angles)
            yield sklearn.utils.shuffle(X_valid, y_valid)

#Define the Model here
def Behav_Clone_Model():
    model = Sequential()
    #Preprocess the incoming image data with a cropping layer
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.))
    
    #1st convolution layer
    model.add(Convolution2D(24,5,5,subsample=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    #2nd Convolution layer
    model.add(Convolution2D(36,5,5,subsample=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    #3rd Convolution layer
    model.add(Convolution2D(48,5,5,subsample=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    #4th Convolution layer
    model.add(Convolution2D(64,3,3,subsample=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    #5th Convolution layer
    model.add(Convolution2D(64,3,3,subsample=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Flatten())
    
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1))
    return model

throwaway_straight_samples = False
nb_ep = 15
samples = []
#Append Samples from already provided data
is_first_row = False
with open('more_d/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #skip the first line as it contains column headers
        if is_first_row == False:
            is_first_row = True
            continue
        samples.append(line)

samples = sklearn.utils.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Print the number of Raw Train and Validation samples
print("Number of Training samples is:",len(train_samples))
print("Number of Validation samples is:",len(validation_samples))

if throwaway_straight_samples == True:
    #function to throwaway some straight driving angles from training samples:
    train_samples = throwaway_func(train_samples)
    print("Number of Training samples after throwaway is:",len(train_samples))
    nb_ep = 15

# compile and train the model using the generator function
train_generator = t_generator(train_samples, batch_size=32)
validation_generator = v_generator(validation_samples, batch_size=32)

#Define the Model
model = Behav_Clone_Model()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= (len(train_samples)*4), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=nb_ep)

model.save('model_comb_fin.h5')
