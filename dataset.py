#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Data set
"""

import numpy as np


class DataSet:

    vectors = []
    labels = []

    def __init__(self, vectors_path, labels_path = None, merge_with = None):
        print("[*] Loading dataset '{}'".format(vectors_path))
        self.loadVectors(vectors_path)
        if merge_with: self.mergeVectors(merge_with)
        if labels_path: self.loadLabels(open(labels_path, "r"))
        if merge_with: self.mergeLabels()
        self._previous_batch = 0


    def loadLabels(self, file_):
        self.labels = np.array([int(line) for line in file_])

    
    def mergeVectors(self, file_path):
        file_ = open(file_path, 'r')
        vectors = [np.array(line.split(','), np.float64) for line in file_]
        vectors = np.transpose(vectors) # Transposed in the file, for some reason...
        vectors /= vectors.max() # Normalise the data
        self.vectors = np.concatenate((self.vectors, vectors))


    def mergeLabels(self):
        labels_file = open("datasets/Validate/valLbls.csv", "r")
        self.labels = np.concatenate((self.labels, np.array([int(line) for line in labels_file])))


    def loadVectors(self, file_path):
        file_ = open(file_path, 'r')
        self.vectors = [np.array(line.split(','), np.float64) for line in file_]
        self.vectors = np.transpose(self.vectors) # Transposed in the file, for some reason...
        self.vectors /= self.vectors.max() # Normalise the data

    
    def shuffle(self):
        assert len(self.vectors) == len(self.labels)
        perm = np.random.permutation(len(self.vectors))
        self.vectors = self.vectors[perm]
        self.labels = self.labels[perm]


    # Return labels under the form [ [ 0 0 0 0 0 0 1 0 0 0 0...  0 0 0], [...], ... ]
    def labels_tensor(self, from_=None, to=None):
        if from_ is None or to is None:
            from_ = 0
            to = len(self.labels)

        tensor = []
        if len(self.labels) == 0:
            tensor = [np.zeros(29)] * len(self.vectors)
        else:
            for label in self.labels[from_:to]:
                tensor.append(np.zeros(len(set(self.labels))))
                tensor[-1][label - 1] = 1

        return tensor


    def nextBatch(self, batch_size):
        from_ = self._previous_batch * batch_size
        to = (self._previous_batch + 1) * batch_size
        self._previous_batch += 1

        if to > len(self.vectors):
            to = len(self.vectors) - 1
            self._previous_batch = 0 # Reset for the next epoch ;)

        batch_labels = self.labels_tensor(from_, to)

        return self.vectors[from_:to], batch_labels
