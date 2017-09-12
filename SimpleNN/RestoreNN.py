##### Simple NN: Working With Restored Network #####     
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

##### Loading Validation Data #####
batchSize = 1000 #This is actually a parameter of the network so make sure it's consistent with whatever batch size the network was trained with.

validationImageDataDict = unpickle('cifar-10-batches-py/data_batch_5')

inputDataValidation = validationImageDataDict[b'data'][:batchSize,:]
outputLabelsValidation = validationImageDataDict[b'labels'][:batchSize]

##### Checking Validation #####
with tf.Session() as sess:
    netSaver = tf.train.import_meta_graph('./NetworkSaves/simpleNN.meta')
    netSaver.restore(sess,tf.train.latest_checkpoint('./NetworkSaves'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0") #tf adds the ":0" to the name. I could not find an explanation but hypothesis that this is to deal with multiple same-named things. Test if bored or curious.
    outputLayer = graph.get_tensor_by_name("outputLayer/BiasAdd:0") 
    values = {x : inputDataValidation}
    outputs = sess.run(outputLayer, feed_dict = values)
    predictedLabels = np.argmax(outputs, 1) #the second argument specifies we want row-wise max indices
    percentCorrect = sum(outputLabelsValidation == predictedLabels)/len(predictedLabels)
    print(predictedLabels)
    print(outputLabelsValidation)

print(percentCorrect)

        
