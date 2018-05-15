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

import os
import csv
import tensorflow as tf

class MultiLayerPerceptron:

    learning_rate = 0.0018
    epochs = 50
    batch_size = 50

    def __init__(self, nb_pixels = 4096, nb_classes = 29, optimizer='Adam'):
        if optimizer == 'SGD': # Stochastic Gradient Descent
            self.optimizerChoice = 'SGD'
        else if optimizer == 'Adam':
            self.optimizerChoice = 'Adam'
        else if optimizer == 'Momentum':
            self.optimizerChoice = 'Momentum'

        self.nb_pixels = nb_pixels
        self.nb_classes = nb_classes
        self.inputs = tf.placeholder(tf.float32, [None, nb_pixels]) # Input neurons
        self.outputs = tf.placeholder(tf.float32, [None, nb_classes]) # output neurons
        self.weights = []
        self.biases = []
        self.hidden_outputs = []


    def setInputs(self, training, validation, testing):
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
            layer_outputs = tf.nn.relu(layer_outputs)

            self.hidden_outputs[layer_index] = layer_outputs

        return tf.add(tf.matmul(self.hidden_outputs[-1], self.weights[-1]), self.biases[-1])


    def train(self):
        # Define loss
        logits = self.createGraph()
        # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.outputs))
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.outputs))

        # Setup the optimizer
        if self.optimizerChoice == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else if self.optimizerChoice == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        else if self.optimizerChoice == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum = 0.75)


        train_op = optimizer.minimize(loss_op)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()


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
                print("Epoch: ", epoch + 1, "cost = ", "{:.12f}".format(avg_cost))

                if epoch != 0 and epoch % 5 == 0 and epoch != self.epochs:
                    pred = tf.nn.softmax(logits)  # Apply softmax to logits
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.outputs, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
                    accuracy = accuracy.eval({self.inputs: self.validation.vectors, self.outputs: self.validation.labels_tensor()})
                    print("Accuracy:", accuracy)

            print("Optimization Finished!")
            # Validation model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.outputs, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
            accuracy = accuracy.eval({self.inputs: self.validation.vectors, self.outputs: self.validation.labels_tensor()})
            print("Accuracy:", accuracy)

            self.session = sess
            self.logits = logits
            usr_input = 'n'
            usr_input = input("Save model? [y/N]: ")

            if usr_input.lower() == 'y':
                # Save the variables to disk.
                save_path = saver.save(sess, "./model.ckpt")
                print("Model saved in path: " + save_path)

            # Testing now
            prediction = tf.argmax(self.logits, 1)
            best = sess.run([prediction], feed_dict={self.inputs: self.testing.vectors})
            self.exportTest(best[0], int(round(accuracy * 100)))


    def exportTest(self, predictions, accuracy):
        with open('testing-{}-percent.csv'.format(accuracy), 'w', newline='') as fp:
            output = csv.writer(fp, delimiter=',')
            data = []
            data.append(['ID', 'Label']) # Header

            for i, label in enumerate(predictions):
                data.append([i + 1, label + 1])

            output.writerows(data)


    def test(self, session = None):
        if session is None:
            session = self.session

        # Testing now
        prediction = tf.argmax(self.logits, 1)
        best = session.run([prediction], feed_dict={self.inputs: self.testing.vectors})
