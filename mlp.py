#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Multi-layer Perceptron
"""

import tensorflow as tf

class MultiLayerPerceptron:

    learning_rate = 0.5
    epochs = 10
    batch_size = 100

    def __init__(self, pixels = 4096, nb_classes = 29):
        self.pixels = pixels
        self.nb_classes = nb_classes
        self.inputs = tf.placeholder(tf.float32, [None, pixels]) # Input neurons
        self.outputs = tf.placeholder(tf.float32, [None, nb_classes]) # output neurons
        self.weights = []
        self.biases = []
        self.hidden_outputs = []


    def setInputs(self, training, testing):
        self.training = training
        self.validation = testing


    def addLayer(self, size):
        if len(self.weights) == 0:
            previous_layer_size = self.pixels
        else:
            _, previous_layer_size = self.weights[-1].shape
            previous_layer_size = previous_layer_size.value

        # From previous layer to new hidden layer
        self.weights.append(tf.Variable(tf.random_normal([previous_layer_size, size], stddev = 0.03), name = "W"))
        self.biases.append(tf.Variable(tf.random_normal([size]), name = "b"))
        self.hidden_outputs.append([])


    def train(self):
        if len(self.hidden_outputs) == 0:
            print("[!] Need at least one hidden layer !")
            exit(1)

        # Connect last layer to output layer
        _, previous_layer_size = self.weights[-1].shape
        self.weights.append(tf.Variable(tf.random_normal([previous_layer_size.value, self.nb_classes], stddev = 0.03), name = "W"))
        self.biases = tf.Variable(tf.random_normal([len(self.training.labels)], name = "b"))

        print("[*] Training for {} epochs...".format(self.epochs))
        # Create the neural network computational graph
        for layer_index, layer_outputs in enumerate(self.hidden_outputs):
            if layer_index == 0:
                previous_layer = self.inputs
            else:
                previous_layer = self.hidden_outputs[layer_index - 1]

            # Feed forward for the current layer
            layer_outputs = tf.add(tf.matmul(previous_layer, self.weights[layer_index]), self.biases[layer_index])
            # Rectified linear unit activation function
            layer_outputs = tf.nn.relu(layer_outputs)
            self.hidden_outputs[layer_index] = layer_outputs

        # Calculate the output layer with the softmax activation function
        # Calculate cross entropy for the optimization (to obtain a cost function)
        outputs = tf.nn.softmax(tf.add(tf.matmul(self.hidden_outputs[-1], self.weights[-1]), self.biases[-1]))
        outputs_clipped = tf.clip_by_value(outputs, 1e-10, 0.9999999)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.outputs * tf.log(outputs_clipped) + (1 - self.outputs) * tf.log(1 - outputs_clipped), axis=1))
        # Setup the optimiser
        # optimiser = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(cross_entropy)
        optimiser = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cross_entropy)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            

        with tf.Session() as sess:
            # Init variables
            sess.run(init_op)
            total_batch = int(len(self.training.labels) / self.batch_size)

            for epoch in range(self.epochs):
                self.training.shuffle()
                avg_cost = 0

                for i in range(total_batch):
                    batch_inputs, batch_outputs = self.training.nextBatch(batch_size = self.batch_size)
                    _, c = sess.run([optimiser, cross_entropy], feed_dict = {self.inputs: batch_inputs, self.outputs: batch_outputs})
                    avg_cost += c / total_batch

                print("Epoch: ", epoch + 1, "cost = ", "{:.3f}".format(avg_cost))

            print(sess.run(accuracy, feed_dict = {self.inputs: self.validation.vectors, self.outputs: self.validation.labels_tensor()}))
