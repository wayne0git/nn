# Ref : https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
def eval_lenet_mnist(model, testData, testLabels):
    (loss, accuracy) = model.evaluate(testData, testLabels)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
