# Ref : https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.optimizers import SGD
from keras import backend as K

class LeNet:
    def build(inputShape, numClasses, activation='relu', weightsPath=None):
        # initialize the model
        model = Sequential()

        # if we are using "channels first", update the inputShape
        # inputShape Argument = (numRows, numCols, numChannels)
        if K.image_data_format() == 'channels_first':
            inputShape = (inputShape[2], inputShape[0], inputShape[1])

        # define the first set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(20, 5, padding='same', input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(50, 5, padding='same', input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # define the second FC layer
        model.add(Dense(numClasses))

        # lastly, define the soft-max classifier
        model.add(Activation('softmax'))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
    
    def train(model, trainData, trainLabels, batch_size=128, epochs=20):
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
        model.fit(trainData, trainLabels, batch_size=batch_size, epochs=epochs)
