#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Convolutional Neural Network
"""

from keras.models import Sequential


class ConvolutionalNeuralNetwork:
    
    learning_rate = 0.01
    epochs = 20
    batch_size = 128
    dropout = 0.75 # Probability to keep units

    def __init__(self, width = 256, height = 256, nb_classes = 29, with_dropout = True):
        self.pixels = width * height
        self.nb_classes = nb_classes
        self.with_dropout = with_dropout
        self.inputs = tf.placeholder(tf.float32, [None, self.pixels])
        self.outputs = tf.placeholder(tf.float32, [None, self.nb_classes])
        self.keep_prob = tf.placeholder(tf.float32) # Dropout

        self.weights = {
            # 5x5 convolution, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 convolution, 32 input, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # Fully connected, 7x7x64 inputs, 1024 outputs
            'wf1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # Outputs, 1024 intputs, nb_classes outputs
            'out': tf.Variable(tf.random_normal([1024, nb_classes]))
        }

        self.biases = {
            'wc1': tf.Variable(tf.random_normal(32)),
            'wc2': tf.Variable(tf.random_normal(64)),
            'wf1': tf.Variable(tf.random_normal(1024)),
            'out': tf.Variable(tf.random_normal(nb_classes))
        }


    def createModel(self, width, height):
        # Reshape the input structure
        self.inputs = tf.reshape(self.inputs, shape=[-1, width, height, 1])
        # Add a convolution layer
        conv1 = conv2d(self.inputs, self.weights['wc1'], self.biases['bc1'])
        # Down sampling with max pooling
        conv1 = maxpool2d(conv1, k=2)

        # Add a convolution layer
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Down sampling with max pooling
        conv2 = maxpool2d(conv2, k=2)

        # Reshape the convolution layer to fit the fully connected layer's input
        fully_connected = tf.reshape(conv2, [-1, self.weights['wf1'].get_shape().as_list()[0])
        fully_connected = tf.add(tf.matmul(fully_connected, self.weights['wf1']), self.biases['bf1'])
        # Apply rectified linear unit activation function
        fully_connected = tf.nn.relu(fully_connected)
        # Apply dropout
        if self.with_dropout:
            fully_connected = tf.nn.dropout(fully_connected, self.dropout)

        # Return final model
        return tf.add(tf.matmul(fully_connected, self.weights['out']), self.biases['out'])


    # Maxpool2D wrapper
    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


    # Conv2D wrapper, with bias and relu activation
    def conv2d(self, x, weights, biases, strides = 1):
        x = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, biases)

        return tf.nn.relu(x)


    def setInputs(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing


    def train(self):
