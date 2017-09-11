##### Simple NN: Building Graph and Training Network #####     
# by Zane Warner
# This project is a fully-connected neural network with 1 hidden layer, implemented to build familiarity with TensorFlow
# It will classify images from the CIFAR-10 dataset
# It is only intended for practice, not to be highly performant
# This module is where I build the graph and train the model
##########

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

##### CIFAR-10 #####
# The CIFAR-10 dataset is described in the report titled:
#    Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
#
# The following code snippet, provided by the CIFAR-10 website, is used to unpack the CIFAR-10 database:
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dataDict = pickle.load(fo, encoding='bytes')
    return dataDict
##########

##### Graph Creation #####
numInputNodes, numHiddenNodes, numOutputNodes = 3072, 5000, 10
batchSize = 10

x = tf.placeholder(tf.float32, shape=(batchSize, numInputNodes), name="x")
y = tf.placeholder(tf.float32, shape=(batchSize, numOutputNodes), name="y")

init = tf.contrib.layers.xavier_initializer() 
#Note that Xavier initializes too low with ReLU since half the input nodes will actually be dead and Xavier assumes they are all in the live regime
#Should really multiply all initial weights by a factor of 2^.5
hiddenLayer = tf.layers.dense(inputs=x, units=numHiddenNodes, activation=tf.nn.relu, kernel_initializer=init, name="hiddenLayer") #dense makes a fully connected layer
outputLayer =  tf.layers.dense(inputs=hiddenLayer, units=numOutputNodes, kernel_initializer=init, name="outputLayer")

loss = tf.losses.softmax_cross_entropy(y, outputLayer)

optimizer = tf.train.GradientDescentOptimizer(1e-6)
updates = optimizer.minimize(loss)

netSaver = tf.train.Saver()

##### Preparing Data #####
imageDataDicts = []

imageDataDicts.append(unpickle('cifar-10-batches-py/data_batch_1'))
imageDataDicts.append(unpickle('cifar-10-batches-py/data_batch_2'))
imageDataDicts.append(unpickle('cifar-10-batches-py/data_batch_3'))
imageDataDicts.append(unpickle('cifar-10-batches-py/data_batch_4'))

#testImageDataDict = unpickle('cifar-10-batches-py/test_batch')

#using the CIFAR datasets presplit batches because I haven't yet chosen to implement adjustable minibatch functionality
numPrebuiltBatches = 4
inputData = {}
outputLabels = {}
outputOneHots = {}
for i in range(numPrebuiltBatches):
    inputData[i] = imageDataDicts[i][b'data']
    outputLabels[i] = imageDataDicts[i][b'labels']
    outputOneHots[i] = np.zeros((len(outputLabels[i]), 10))
    for example in outputLabels[i]:
        outputOneHots[i][example, outputLabels[i][example]] = 1
    inputData[i] = inputData[i][:batchSize,:]
    outputOneHots[i] = outputOneHots[i][:batchSize,:]

##### Training Network #####
numEpochs=1
lossValue = np.zeros(numEpochs)
lossValueValidation = np.zeros(numEpochs)
batchOrderer = np.arange(numPrebuiltBatches)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(numEpochs):
        np.random.shuffle(batchOrderer)
        for i in batchOrderer:
            values = {x : inputData[i],
                      y : outputOneHots[i]}
            batchLossValue, throwaway = sess.run([loss, updates], feed_dict = values)
            lossValue[epoch] += batchLossValue
        lossValue[epoch] = lossValue[epoch]/numPrebuiltBatches
        print('Epoch {}--Loss Value: {}'.format(epoch+1, lossValue[epoch]))
    netSaver.save(sess, './NetworkSaves/simpleNN')

plt.plot(lossValue)
plt.show()