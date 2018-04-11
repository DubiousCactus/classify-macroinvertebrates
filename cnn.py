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

import tensorflow as tf


class ConvolutionalNeuralNetwork:
    
    learning_rate = 0.01
    epochs = 20
    batch_size = 128
    dropout = 0.4

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
    def conv2d(self, x, weights, biases, strides = 2):
        x = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, biases)

        return tf.nn.relu(x)


    def setInputs(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing


    def train(self):
         # Because Dropout has different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
        logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

         # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32))
        ) 
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op}
        )

        # Build the estimator
        model = tf.estimator.Estimator(estim_specs)
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': mnist.train.images}, y=mnist.train.labels,
            batch_size=batch_size, num_epochs=None, shuffle=True)
        # Train the Model
        model.train(input_fn, steps=num_steps)

        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': mnist.test.images}, y=mnist.test.labels,
            batch_size=batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = model.evaluate(input_fn)

        print("Testing Accuracy:", e['accuracy'])
