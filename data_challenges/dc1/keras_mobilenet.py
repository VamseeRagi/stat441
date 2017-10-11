import numpy as np
import pandas as pd
import tensorflow as tf

# Loading the training set
training_set = pd.read_csv("fashion-mnist_train.csv", dtype='int')

# Loading the training data
features = training_set.columns[1:]
X_train = training_set.drop('label', axis = 1)
Y_train = training_set['label']

# Preprocessing data: doubling the dimensions since they are 28px (< 32)
#   and hence do not have valid input shape for the MobileNet model

