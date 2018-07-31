Food Item Classifier (Alexnet) : Classification of food image uploaded by user for an online food ordering application using Alexnet from scratch.
==================

There are lots of food images uploaded by user of an online food ordering app. So instead of manually categorizing, We will use the deep learning for auto classification.

We will have 101 types of food items and we will predict any new image of food item uploaded by user of app.

We are not using here any pretrained model for our classification task, but we have built an alexnet archicture from scratch using Keras python library.

The data has been reformatted as HDF5 and specifically Keras HDF5Matrix which allows them to be easily read in. The file names indicate the contents of the file.

* food_c101_n1000_r384x384x3.h5 means there are 101 categories represented, with n=1000 images, that have a resolution of 384x384x3 (RGB, uint8)

* food_test_c101_n1000_r32x32x1.h5 means the data is part of the validation set, has 101 categories represented, with n=1000 images, that have a resolution of 32x32x1 (float32 from -1 to 1)

We are visulizing accuracy and loss of validation after each epoch using matplotlib library.

Installation
==================

To start with project just follow the few steps 

	$ git clone https://github.com/keyurr2/shape-classifier-cnn.git
	$ pip install -r requirements.txt
	
This will install python libraries required to start with Deep Learning like Tensorflow and Keras

NOTE: We are using Python 3 in this project.


How to run this project
==================================================
The first step is to train the model using training dataset.
	
	$ python cnn.py

It will take some time as we are using alexnet and it's 101 classes.

Once the model is trained you can find the saved model and use that model to classify new images for your online food ordering application.

You can train model for your own dataset also and try to play with that.

You can find the confusion matrix and accuracy of model after prediction.

Authors
==================

* **Keyur Rathod (keyur.rathod1993@gmail.com)**

License
==================

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
