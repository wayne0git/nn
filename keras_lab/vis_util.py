import cv2
import numpy as np

from keras import backend as K
from matplotlib import pyplot as plt

# Ref : https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
def vis_lenet_mnist(model, testData, testLabels):
    # randomly select a testing digit
    i = np.random.choice(np.arange(0, len(testLabels)), size=(1,))[0]

    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # extract the image
    if K.image_data_format() == 'channels_first':
        image = (testData[i][0] * 255).astype('uint8')
    else:
        image = (testData[i] * 255).astype('uint8')

    # merge the channels into one image
    image = cv2.merge([image] * 3)

    # resize for better visualization
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
 
    # show the image and prediction
    cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(testLabels[i])))
    plt.figure()
    plt.imshow(image)
