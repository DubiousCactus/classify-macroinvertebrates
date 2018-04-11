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

from sklearn import svm


class SupportVectorMachine:

    def __init__(self):
        self.model = svm.SVC(kernel='rbf', C=1, gamma='auto')
        

    def setInputs(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing


    def train(self):
        self.model.fit(self.training.vectors, self.training.labels)
        print(self.model.score(self.validation.vectors, self.validation.labels))


    def test(self):
        print(self.model.predict(self.testing.vectors))
