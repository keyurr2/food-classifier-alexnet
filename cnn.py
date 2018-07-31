#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 18:35:30 2018

@author: keyur-r
"""

# Building architecture of our CNN classifier
import numpy as np
import pandas as pd
import keras
from keras.optimizers import Adam
from model import cnn_model
from alexnet import AlexNet

# Variables
CLASSES = 101
IMAGE_SIZE = 64
CHANNELS = 3
NUM_EPOCH = 500
LEARN_RATE = 1.0e-4
BATCH_SIZE = 32

# Model Architecture and Compilation
model = AlexNet(CLASSES, IMAGE_SIZE, CHANNELS)
adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(
    optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# Image Preprocessing

from get_train_test import read_from_h5

base_path = "dataset"
train_h5_file = "food_c101_n10099_r64x64x3.h5"
test_h5_file = "food_test_c101_n1000_r64x64x3.h5"

X_train, y_train, X_test, y_test = read_from_h5(
    base_path, train_h5_file, test_h5_file)

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.h5', monitor='val_loss', save_best_only=True, mode='auto')
steps_per_epoch = int(len(y_train) / BATCH_SIZE)  # 300
validation_steps = int(len(y_test) / BATCH_SIZE)  # 90


# Training
model_info = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                       batch_size=BATCH_SIZE, shuffle="batch", epochs=NUM_EPOCH, verbose=1)

model.save("food_classification.h5")

# plot model history after each epoch
from visulization import plot_model_history
plot_model_history(model_info)
