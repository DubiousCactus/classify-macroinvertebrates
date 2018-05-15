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
from sklearn.externals import joblib


class SIFT_SupportVectorMachine:

    def __init__(self, K = 405):
        random.seed()
        # self.model = svm.SVC(probability=False, kernel='rbf', C=8, gamma=.0082)
        self.model = svm.SVC(probability=False, kernel='linear', C=8, gamma=.0082)
        self.training_descriptors = {}
        self.validation_descriptors = {}
        self.testing_descriptors = {}
        self.training_histograms = {}
        self.validation_histograms = {}
        self.kmeans = None
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


    def createHistograms(self, dataset, descriptors, cluster):
        histograms = {}
        labels = []
        for index, image in sorted(dataset.images_paths.items()):
            # For each descriptor of the current image
            histograms[index] = np.zeros(self.n_clusters)
            # Hopefully the labels order will be the same as the images
            # order... Or will it...
            labels.append(dataset.labels[index])
            if descriptors[index] is None: continue
            for descriptor in descriptors[index]:
                prediction = cluster.predict([descriptor])
                histograms[index][prediction] += 1

        return (histograms, labels)


    def clusterFeatures(self):
        print("[*] Clustering feature descriptors...")
        # Concatenate training feature descriptors
        feature_descriptors = []
        # Use .item() when loading the features from npy files
        # but not when they're in the RAM already
        for image_descriptors in list(self.training_descriptors.item().values()):
            if image_descriptors is None: continue # Why are some of them None??
            for image_descriptor in image_descriptors:
                feature_descriptors.append(image_descriptor)

        filename = "kmeans_classifier.pkl"
        if os.path.isfile(filename):
            print("\t-> Restoring K-Means classifier from '{}'".format(filename))
            self.kmeans = joblib.load(filename)
        else:
            print("\t-> Running K-Means with K={}".format(self.n_clusters))
            self.kmeans = cluster.KMeans(n_clusters =
                                    self.n_clusters, n_jobs=-1).fit(feature_descriptors)
            print("[*] Saving classifier as '{}'...".format(filename))
            _ = joblib.dump(self.kmeans, filename, compress=9)

        if os.path.isdir('histograms'):
            print("[*] Restoring histograms from '{}'".format('histograms/'))
            self.training_histograms = np.load('histograms/training_histograms.npy')
            self.training_histograms_labels = np.load('histograms/training_labels.npy')
            self.validation_histograms = np.load('histograms/validation_histograms.npy')
            self.validation_histograms_labels = np.load('histograms/validation_labels.npy')
        else:
            os.mkdir('histograms')
            print("[*] Creating histograms...")
            print("\t-> Training samples")
            hist, labels = self.createHistograms(
                self.training,
                self.training_descriptors.item(),
                self.kmeans
            )
            self.training_histograms = hist
            self.training_histograms_labels = labels
            np.save('histograms/training_histograms.npy', self.training_histograms)
            np.save('histograms/training_labels.npy', self.training_histograms_labels)

            print("\t-> Validation samples")
            hist, labels = self.createHistograms(
                self.validation,
                self.validation_descriptors.item(),
                self.kmeans
            )
            self.validation_histograms = hist
            self.validation_histograms_labels = labels
            np.save('histograms/validation_histograms.npy', self.validation_histograms)
            np.save('histograms/validation_labels.npy', self.validation_histograms_labels)


    # Return the max amount of features per image and the approximate amount of features per image
    def featuresPerImage(self):
        totalFeaturesInTraining = 0
        maxFeaturesInTraining = 0
        nbImages = 0
        for image_descriptors in list(self.training_descriptors.item().values()):
            if image_descriptors is None: continue
            totalFeaturesInTraining += len(image_descriptors)
            if len(image_descriptors) > maxFeaturesInTraining:
                maxFeaturesInTraining = len(image_descriptors)
            nbImages += 1

        return (
            maxFeaturesInTraining, round(totalFeaturesInTraining / nbImages)
        )


    def loadFeatures(self, path):
        print("[*] Loading SIFT features...")
        self.training_descriptors = np.load(path + 'training_descriptors.npy')
        self.validation_descriptors = np.load(path + 'validation_descriptors.npy')
        self.testing_descriptors = np.load(path + 'testing_descriptors.npy')
        maxFeatures, avrgFeatures = self.featuresPerImage()
        print("[*] Approx. SIFT features per image: {}".format(avrgFeatures))
        print("[*] Max SIFT features per image: {}".format(maxFeatures))


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
        # print("[*] Shuffling training histograms...")
        # self.training_descriptors.shuffle()
        print("[*] Fitting the Support Vector Machine...")
        self.model.fit(list(self.training_histograms.item().values()),
                       self.training_histograms_labels)
        print("[*] Classification score: {:.6f}%".format(
            self.model.score(list(self.validation_histograms.item().values()),
                             self.validation_histograms_labels) * 100)
        )


    def test(self):
        print("[*] Classifying test samples...")
        testing_histograms, testing_labels = self.createHistograms(self.testing, self.testing_descriptors, self.kmeans)
        predictions = self.model.predict(list(self.testing_histograms.values()),
                           testing_labels)
        self.exportTest(predictions)



    def exportTest(self, predictions):
        with open('testing-SIFT_SVM.csv', 'w', newline='') as fp:
            output = csv.writer(fp, delimiter=',')
            data = []
            data.append(['ID', 'Label'])  # Header

            for i, label in enumerate(predictions):
                data.append([i + 1, label])

            output.writerows(data)
