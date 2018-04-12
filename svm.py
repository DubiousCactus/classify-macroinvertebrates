#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@translaptop>
#
# Distributed under terms of the MIT license.

"""
Support Vector Machine
"""

from sklearn import svm, metrics


class SupportVectorMachine:

    def __init__(self):
        self.model = svm.SVC(probability=False, kernel='rbf', C=19, gamma=.0082)
        

    def setInputs(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing


    def train(self):
        print("[*] Shuffling training samples...")
        self.training.shuffle()
        print("[*] Fitting the Support Vector Machine...")
        self.model.fit(self.training.vectors, self.training.labels)
        print("[*] Classification score: {:.2f}%".format(self.model.score(self.validation.vectors, self.validation.labels)*100))


    def test(self):
        # print(self.model.predict(self.testing.vectors))
        pass
