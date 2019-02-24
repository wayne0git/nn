import keras.datasets as k_dataset

from keras import backend as K
from keras.utils import np_utils

# Ref : https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
def load_mnist():
    # Load data from Keras dataset
    ((trainData, trainLabels), (testData, testLabels)) = k_dataset.mnist.load_data()

    # Reshape data
    if K.image_data_format() == 'channels_first':
    	trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    	testData = testData.reshape((testData.shape[0], 1, 28, 28))
    else:
    	trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    	testData = testData.reshape((testData.shape[0], 28, 28, 1))

    # Normalize
    trainData = trainData.astype('float32') / 255.0
    testData = testData.astype('float32') / 255.0

    # Convert to categorical label
    trainLabels = np_utils.to_categorical(trainLabels, 10)
    testLabels = np_utils.to_categorical(testLabels, 10)

    return (trainData, trainLabels), (testData, testLabels)
