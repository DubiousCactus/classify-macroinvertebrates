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

import os
import cv2
import random
import numpy as np

from sklearn import svm, cluster


class SIFT_SupportVectorMachine:

    def __init__(self, K = 80):
        random.seed()
        self.model = svm.SVC(probability=False, kernel='rbf', C=19, gamma=.0082)
        self.training_descriptors = {}
        self.validation_descriptors = {}
        self.testing_descriptors = {}
        self.n_clusters = K


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
            key_points, feature_descs = sift.detectAndCompute(image, None)
            descriptors[index] = feature_descs

        return descriptors


    def saveFeatureDescriptors(self, descriptors, name):
        print("\t\t* Saving feature descriptors as '{}.npy'...".format(name))
        if not os.path.isdir('sift_features'):
            os.mkdir('sift_features')

        np.save('sift_features/{}.npy'.format(name), descriptors)


    def clusterFeatures(self):
        print("[*] Clustering feature descriptors...")
        # Concatenate training feature descriptors
        print(self.training_descriptors)
        feature_descriptors = self.training_descriptors.items()
        kmeans = cluster.KMeans(n_clusters =
                                self.n_clusters).fit(feature_descriptors)
        print("[*] Creating histograms...")
        for index, image in self.training.images_paths.items():
            clusters = kmeans.predict(self.training_descriptors[index])
            printf("[INFO] Image {} cluster associations:".format(index))
            printf(clusters[0])
            # For each descriptor of the current image
            for descriptor in self.training_descriptors[index]:
                print(clusters.predict(descriptor))
            break


    def loadFeatures(self, path):
        print("[*] Loading SIFT features...")
        self.training_descriptors = np.load(path + 'training_descriptors.npy')
        self.validation_descriptors = np.load(path + 'validation_descriptors.npy')
        self.testing_descriptors = np.load(path + 'testing_descriptors.npy')


    def extractAndSaveFeatures(self):
        print("[*] Extracting SIFT features...")
        print("\t-> From training data")
        self.training_descriptors = self.computeSIFTFeatures(self.training)
        self.saveFeatureDescriptors(self.training_descriptors,
                                    "training_descriptors")
        print("\t-> From validation data")
        self.validation_descriptors = self.computeSIFTFeatures(self.validation)
        self.saveFeatureDescriptors(self.validation_descriptors,
                                    "validation_descriptors")
        print("\t-> From testing data")
        self.testing_descriptors = self.computeSIFTFeatures(self.testing)
        self.saveFeatureDescriptors(self.testing_descriptors, "testing_descriptors")


    def train(self):
        if os.path.isdir('sift_features'):
            self.loadFeatures('sift_features/')
        else:
            self.extractAndSaveFeatures()

        self.clusterFeatures()
