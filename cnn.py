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

    def __init__(self, img_x = , img_y = , nb_classes = 29):
        self.pixels = img_x * img_y
        self.nb_classes = nb_classes
        input_shape = (img_x, img_y, 1) # The tensor shape
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(64, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dense(nb_classes, activation='softmax'))


    def setInputs(self, training, testing):
        self.training = training
        self.validation = testing


    def train(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=self.learning_rate),
              metrics=['accuracy'])
        self.model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history]
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
