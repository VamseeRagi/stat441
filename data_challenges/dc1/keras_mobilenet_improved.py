import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import misc

from keras.applications.mobilenet import MobileNet
from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K

# Loading the training data as numpy arrays
training_set = pd.read_csv("fashion-mnist_train.csv", dtype='int')
X_train = training_set.drop('label', axis = 1).as_matrix()
Y_train = training_set['label'].as_matrix()

dim = 28


# Preprocessing data: doubling the dimensions since they are 28px (< 32)
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
    resized_images = [1.0*misc.imresize(img.reshape(height, width),
                                          (height_x2, width_x2))
                      for img in images]

    return np.array(resized_images) / 255


X_train = double_dimensions(X_train, dim)

# Data
## Training the model

# The Input Layer: accepts 56 by 56 images
input_image = Input(shape=(dim*2,dim*2))
# Adding image channels to data
input_image_dim = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,3),3,3))(input_image)
# Using MobileNet as the pre-trained base model
base_model = MobileNet(input_tensor=input_image_dim, include_top=False, pooling='avg')
# Randomly dropping 1/2 of the input units to prevent overfitting
output = Dropout(0.5)(base_model.output)
# Adding a logistic layer for 10 classes
predict = Dense(10, activation='softmax')(output)

# the final model to train
model = Model(inputs=input_image, outputs=predict)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.summary()

model.fit(X_train, Y_train, batch_size=200, epochs=50)

# Predicting labels on the test set
test_set = pd.read_csv("test_data.csv", index_col=0, dtype='int')
X_test = test_set.as_matrix()
X_test = double_dimensions(X_test, dim)

Y_probs = model.predict(X_test)
Y_predicted = Y_probs.argmax(axis=-1)

# Creating the submission
mn_sel_submit = pd.DataFrame({'ids': [i for i in range(10000)],
                             'label': Y_predicted}, dtype=int)
mn_sel_submit.to_csv('improved_mn_predictions.csv', index=False)