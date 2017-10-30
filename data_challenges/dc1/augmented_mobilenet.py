import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt

#from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K

# Loading the training data as numpy arrays
training_set = pd.read_csv("fashion-mnist_train.csv", dtype='int')
X_train = training_set.drop('label', axis = 1).as_matrix()
Y_train = training_set['label'].as_matrix()

dim = 28

# Checking if the images are greyscale
# sample = X_train[200].reshape(28,28)
# plt.imshow(sample)
# plt.show()

# Preprocessing and augmenting data

# Doubling the dimensions since they are 28px (< 32)
#   and hence do not have valid input shape for the MobileNet model
def double_dimensions(images, height, width=-1):
    '''
    double_dimensions(images, height, width) doubles the size
        of the image data in the numpy array images
    height and width must be the current dimensions of the images
    If width is omitted, the data is treated as square images, i.e.
        width = height
    '''
    if width == -1:
        width = height

    # reshaping data to actual image dimensions and doubling the size
    width_x2 = width * 2
    height_x2 = height * 2
    resized_images = [1.0*misc.imresize(img.reshape(height, width), (height_x2, width_x2))
                      for img in images]
    # Adding another dimension for image channel
    #greyscale_images = [img.reshape((56, 56, -1)) for img in resized_images]
    # transforming data from [0,255] to [0,1]
    return np.array(resized_images)/255

X_train = double_dimensions(X_train, dim)


def datagen(X, Y, batch_size):
    # standardize the data
    X_mean = np.mean(X, axis=0)
    X_sigma = np.std(X, axis=0)
    X = (X - X_mean) / (X_sigma + 1e-7)

    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        batch_x = []
        batch_y = []

        for i in range(len(X)):
            aug_X = X[i]
            # random horizontal flips
            if np.random.random() < 0.5:
                aug_X = aug_X[::-1]

            batch_x.append(aug_X)
            batch_y.append(Y[i])

            if len(batch_x) == batch_size:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []

        if batch_x:
            yield np.array(batch_x), np.array(batch_y)
            batch_x = []
            batch_y = []

# Training the model

# The Input Layer: accepts 56 by 56 images
input_image = Input(shape=(dim * 2, dim * 2))

# Expanding the dimensionality of the input data to add image channel
input_image_dim = Lambda(
    lambda x: K.repeat_elements(K.expand_dims(x, 3), 3, 3))(input_image)
# Using MobileNet as the pre-trained base model
base_model = MobileNet(input_tensor=input_image_dim, include_top=False,
                       pooling='avg')
# Randomly dropping 1/2 of the input units to prevent overfitting
output = Dropout(0.5)(base_model.output)
# Adding a logistic layer for 10 classes
predict = Dense(10, activation='softmax')(output)

# the final model to train
model = Model(inputs=input_image, outputs=predict)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#model.summary()

model.fit_generator(datagen(X_train, Y_train, batch_size=200),
                    steps_per_epoch=len(X_train) / 200, epochs=10)

# Predicting labels on the test set
test_set = pd.read_csv("test_data.csv", index_col=0, dtype='int')
X_test = test_set.as_matrix()
X_test = double_dimensions(X_test, dim)

# Generating the probabilities of the data being in each class
Y_probs = model.predict(X_test)

# Labelling the data by taking the highest probablility class
Y_predicted = Y_probs.argmax(axis=-1)