#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@translaptop>
#
# Distributed under terms of the MIT license.

"""
SIFT features-based Support Vector Machine
"""

import cv2
import numpy as np
from sklearn import svm

class SIFT_SupportVectorMachine:

    def __init__(self):
        # self.model = svm.SVC(probability=False, kernel='rbf', C=19, gamma=.0082)
        self.training_descriptors = {}
        self.validation_descriptors = {}
        self.testing_descriptors = {}


    def setInputs(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing

    
    def computeSIFTFeatures(self, dataset):
        descriptors = {}
        cv2.CV_LOAD_IMAGE_COLOR = 0
        for index, image in dataset.images_paths.items():
            image = cv2.imread(image, cv2.CV_LOAD_IMAGE_COLOR)
            sift = cv2.xfeatures2d.SIFT_create()
            key_points, feature_desc = sift.detectAndCompute(image, None)
            descriptors[index] = feature_desc

        return descriptors


    def saveFeatureDescriptors(self, descriptors):
        print("\t\t* Saving feature descriptors...")
        np.save('training_features.npy', descriptors)


    def clusterFeatures(self):
        print("[*] Clustering feature descriptors...")
        # Concatenate training feature descriptors
        # for feature_desc in self.training_descriptors:

        



    def train(self):
        print("[*] Extracting SIFT features...")
        print("\t-> From training data")
        training_descriptors = self.computeSIFTFeatures(self.training)
        self.saveFeatureDescriptors(training_descriptors)
        print("\t-> From validation data")
        validation_descriptors = self.computeSIFTFeatures(self.validation)
        self.saveFeatureDescriptors(validation_descriptors)
        print("\t-> From testing data")
        testing_descriptors = self.computeSIFTFeatures(self.testing)
        self.saveFeatureDescriptors(testing_descriptors)
        self.clusterFeatures()
