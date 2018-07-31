#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:37:31 2018

@author: keyur-r
"""

# Imaage Preprocessing
from keras.utils.io_utils import HDF5Matrix
from keras.preprocessing.image import ImageDataGenerator
import os

def read_from_h5(base_path, train_h5_file, test_h5_file):
    train_h5_path = os.path.join(base_path, train_h5_file)
    test_h5_path = os.path.join(base_path, train_h5_file)
    
    X_train = HDF5Matrix(train_h5_path, 'images')
    y_train = HDF5Matrix(train_h5_path, 'category')
    print('In Data',X_train.shape,'=>', y_train.shape)
    
    X_test = HDF5Matrix(test_h5_path, 'images')
    y_test = HDF5Matrix(test_h5_path, 'category')
    print('In Data',X_test.shape,'=>', y_test.shape)
    return X_train, y_train, X_test, y_test


def read_from_directory(train_dir, test_dir, input_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    training_set = train_datagen.flow_from_directory(
        train_dir, target_size=(input_size, input_size), batch_size=batch_size, class_mode='categorical')
    #X_images, y_labels = training_set.filenames, training_set.classes
    test_set = test_datagen.flow_from_directory(
        test_dir, target_size=(input_size, input_size), batch_size=batch_size, class_mode='categorical')
    return training_set, test_set