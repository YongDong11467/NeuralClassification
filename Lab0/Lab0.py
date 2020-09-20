
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import pandas as pd
from pandas import DataFrame
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
            return 1/(1+np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        f = self.__sigmoid(x)
        return f * (1 - f)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    # original epochs = 100000 mbs = 100
    def train(self, xVals, yVals, epochs = 5, minibatches = True, mbs = 100):
        for i in range(0, epochs):
            print(i)
            x_generator = self.__batchGenerator(xVals, mbs)
            y_generator = self.__batchGenerator(yVals, mbs)
            for j in range(0, int(len(xVals) / mbs)):
                xVal = next(x_generator)
                yVal = next(y_generator)
                layer1out, layer2out = self.__forward(xVal)
                l2e = (yVal - layer2out)
                l2Delta = l2e * self.__sigmoidDerivative(layer2out)
                l1e = np.dot(l2Delta, self.W2.T)
                l1Delta = l1e * self.__sigmoidDerivative(layer1out)
                self.W1 += np.dot(xVal.T, l1Delta) * self.lr
                self.W2 += np.dot(layer1out.T, l2Delta) * self.lr

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        modout = np.zeros(layer2.shape)
        for i in range(0, layer2.shape[0]):
            modout[i][layer2[i].argmax(axis=0)] = 1
        return modout



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain, xTest = xTrain / 255, xTest / 255
    xTrain = np.ndarray.flatten(xTrain).reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])
    xTest = np.ndarray.flatten(xTest).reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        #TODO: Write code to build and train your custom neural net.
        model = NeuralNetwork_2Layer(xTrain.shape[1], yTrain.shape[1], 512)
        model.train(xTrain, yTrain)
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        #TODO: Write code to build and train your keras neural net.

        # model = tf.keras.models.Sequential(
        #     [tf.keras.layers.Flatten(), tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
        # tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(784, activation=tf.nn.sigmoid), tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # process y
        (xTrain2, yTrain2), (xtest2, ytest2) = tf.keras.datasets.mnist.load_data() # this line here just for testing

        model.fit(xTrain, yTrain, epochs=5)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        #TODO: Write code to run your custon neural net.
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        #TODO: Write code to run your keras neural net.
        return model.evaluate(data)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    table = dict.fromkeys(range(10), 0)
    acc = 0
    predsmod = np.zeros(preds.shape[0])
    yTestmod = np.zeros(yTest.shape[0])
    for i in range(preds.shape[0]):
        predsmod[i] = preds[i].argmax(axis=0)
        yTestmod[i] = yTest[i].argmax(axis=0)
        # Add to dictionary
        table[preds[i].argmax(axis=0)] += 1
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    #F1 = 2 * ((precision * recall) / (precision + recall))
    #precision = acc / acc + FP
    #recall = acc / acc + FN
    print(confusion_matrix(yTestmod, predsmod))
    print(classification_report(yTestmod, predsmod))
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>==============================preds[i]==================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

    #Quiz
    # (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    # xTrain, xTest = xTrain / 255.0, xTest / 255
    # model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
    #                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(xTrain, yTrain, epochs=5)
    # print("loss: %f\naccuracy: %f" % tuple(model.evaluate(xTest, yTest)))



if __name__ == '__main__':
    main()
