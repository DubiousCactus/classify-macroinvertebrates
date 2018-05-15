#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@translaptop>
#
# Distributed under terms of the MIT license.

"""
Nearest Neighbour
"""

import csv
import numpy as np


class NearestNeighbour:


    def __init__(self):
        return

    def setInputs(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing


    def accuracy(self, computed_labels, original_labels):
        matches = 0
        for index, computed_label in enumerate(computed_labels):
            if computed_label == original_labels[index]:
                matches += 1

        return matches / len(original_labels)


    def validate(self):
        print("[*] Computing validation score...")
        validation = []
        for val_sample in self.validation.vectors:
            candidate = {}
            candidate['distance'] = np.linalg.norm(val_sample - self.training.vectors[0])
            candidate['label'] = self.training.labels[0]
            for index, training_sample in enumerate(self.training.vectors[1:]):
                distance = np.linalg.norm(val_sample - training_sample)
                if distance < candidate['distance']:
                    candidate['distance'] = distance
                    candidate['label'] = self.training.labels[index]

            validation.append(candidate['label'])

        print("[*] Validation accuracy: {}%".format(
            self.accuracy(validation, self.validation.labels) * 100)
        )
