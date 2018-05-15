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

import csv

from sklearn import svm


class SupportVectorMachine:

    def __init__(self):
        # self.model = svm.SVC(kernel='rbf', C=19, gamma=.008195)
        # self.model = svm.SVC(probability=False, kernel='linear', C=19, gamma=.008195)
        self.model = svm.SVC(kernel='linear', C=0.98999)


    def setInputs(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing


    def train(self):
        print("[*] Shuffling training samples...")
        self.training.shuffle()
        print("[*] Fitting the Support Vector Machine...")
        self.model.fit(self.training.vectors, self.training.labels)
        print("[*] Classification score: {:.6f}%".format(
            self.model.score(self.validation.vectors, self.validation.labels) * 100)
        )


    def test(self):
        print("[*] Classifying test samples...")
        predictions = self.model.predict(self.testing.vectors)
        self.exportTest(predictions)


    def exportTest(self, predictions):
        with open('testing-SVM.csv', 'w', newline='') as fp:
            output = csv.writer(fp, delimiter=',')
            data = []
            data.append(['ID', 'Label'])  # Header

            for i, label in enumerate(predictions):
                data.append([i + 1, label])

            output.writerows(data)
