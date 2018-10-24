import cv2
import tensorflow as tf
import os
import glob
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import model_from_json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import theano
import matplotlib.pyplot as plt
import matplotlib

# specifiying the training set directory
training_dir = './training_data'
testing_dir = './testing_data'
img_size = 128
img_channels = 3
listing = sorted(os.listdir(training_dir))
num_classess = size(listing)
print("No of Classes: %d".format(num_classes))
files = glob.glob(training_dir)
# making the images list
images = []
for file in files:
	image = cv2.resize(image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    images.append(image)

# making the labels
labels = np.zeros(len(num_classes))
labels = ['apple', 'orange']

# shuffling the images and the labels together
images, labels = shuffle(images, labels, random_state = 4)

batch_size = 25

num_epoch = 100

num_filter1 = 35
num_filter2 = 35

num_pool = 2
num_conv = 5

(x,y) = (images,labels)

# Split x and y into training and testing sets in random order
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# Assigning X_train and X_test as float
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

# Normalization of data 
# Data pixels are between 0 and 1
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#implement a model
model = Sequential()

model.add(Convolution2D(num_filters1, kernel_size = (num_conv, num_conv), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool), strides = (2, 2)))

model.add(Convolution2D(num_filters2, kernel_size = (num_conv, num_conv), activation='relu'))
model.add(MaxPooling2D(pool_size = (num_pool, num_pool), strides = (2, 2)))

model.add(Flatten())
model.add(Dense(500, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))

optimiser = SGD(lr = 0.01)
model.compile(loss = 'categorical_crossentropy', optimiser = optimiser, metrics = ['accuracy'])

hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = num_epoch, verbose = 1, validation_split = 0.25)

# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss'] 
train_acc = (hist.history['acc'])
val_acc = (hist.history['val_acc'])

score = model.evaluate(X_test, Y_test, verbose=0) # accuracy check
print('Test accuracy:', score[1]) # Prints test accuracy
 
y_pred = model.predict_classes(X_test) # Predicts classes of all images in test data 

p = model.predict_proba(X_test) # To predict probability

print('\nConfusion Matrix')
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred)) # Prints Confusion matrix for analysis

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to H5
model.save_weights("model.h5")
print("Saved model to disk")

# X_test and Y_test are saved so model can be tested 
np.save('X_test', X_test)
np.save('Y_test', Y_test)