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

    learning_rate = 0.05
    epochs = 15
    batch_size = 20

    def __init__(self, nb_pixels = 4096, nb_classes = 29):
        self.nb_pixels = nb_pixels
        self.nb_classes = nb_classes
        self.inputs = tf.placeholder(tf.float32, [None, nb_pixels]) # Input neurons
        self.outputs = tf.placeholder(tf.float32, [None, nb_classes]) # output neurons
        self.weights = []
        self.biases = []
        self.hidden_outputs = []


    def setInputs(self, training, validation, testing = None):
        self.training = training
        self.validation = validation
        self.testing = testing


    def addLayer(self, size):
        if len(self.weights) == 0:
            previous_layer_size = self.nb_pixels
        else:
            _, previous_layer_size = self.weights[-1].shape
            previous_layer_size = previous_layer_size.value

        # From previous layer to new hidden layer
        self.weights.append(tf.Variable(tf.random_normal([previous_layer_size, size], stddev = 0.03)))
        self.biases.append(tf.Variable(tf.random_normal([size])))
        self.hidden_outputs.append([])


    def createGraph(self):
        if len(self.hidden_outputs) == 0:
            print("[!] Need at least one hidden layer !")
            exit(1)

        # Connect last layer to output layer
        _, previous_layer_size = self.weights[-1].shape
        self.weights.append(tf.Variable(tf.random_normal([previous_layer_size.value, self.nb_classes], stddev = 0.03)))
        self.biases = tf.Variable(tf.random_normal([len(self.training.labels)]))

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
            # layer_outputs = tf.nn.relu(layer_outputs)
            self.hidden_outputs[layer_index] = layer_outputs

        return tf.matmul(self.hidden_outputs[-1], self.weights[-1]) + self.biases[-1]


    def train(self):
        # Define loss
        logits = self.createGraph()
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.outputs))

        # Setup the optimizer
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            # Init variables
            sess.run(init_op)
            total_batch = int(len(self.training.vectors) / self.batch_size)

            for epoch in range(self.epochs):
                self.training.shuffle()
                avg_cost = 0

                for i in range(total_batch):
                    batch_inputs, batch_outputs = self.training.nextBatch(self.batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict = {self.inputs: batch_inputs, self.outputs: batch_outputs})
                    avg_cost += c / total_batch

                print("Epoch: ", epoch + 1, "cost = ", "{:.9f}".format(avg_cost))
            print("Optimization Finished!")
            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.outputs, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
            print("Accuracy:", accuracy.eval({self.inputs: self.validation.vectors, self.outputs: self.validation.labels_tensor()}))
