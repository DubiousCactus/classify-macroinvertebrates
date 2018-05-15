#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@translaptop>
#
# Distributed under terms of the MIT license.

"""
SIFT-based Nearest Neighbour
"""

import os
import cv2
import random
import numpy as np

from sklearn import cluster
from sklearn.externals import joblib

class SIFT_NearestNeighbour:

    def __init__(self, K = 405):
        random.seed()
        self.training_descriptors = {}
        self.validation_descriptors = {}
        self.testing_descriptors = {}
        self.training_histograms = {}
        self.validation_histograms = {}
        self.kmeans = None
        self.n_clusters = K


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


    def setInputs(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing


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


    def accuracy(self, computed_labels, original_labels):
        matches = 0
        for index, computed_label in enumerate(computed_labels):
            if computed_label == original_labels[index]:
                matches += 1

        return matches / len(original_labels)


    def validate(self):
        print("[*] Computing validation score...")
        if os.path.isdir('sift_features'):
            self.loadFeatures('sift_features/')
        else:
            self.extractAndSaveFeatures()

        self.clusterFeatures()

        validation = []
        for index, val_hist in enumerate(list(self.validation_histograms.item().values())):
            candidate = {}
            candidate['distance'] = np.linalg.norm(
                val_hist - list(self.training_histograms.item().values())[0]
            )
            candidate['label'] = self.training_histograms_labels[0]
            for index, training_sample in enumerate(list(self.training_histograms.item().values())[1:]):
                distance = np.linalg.norm(val_hist - training_sample)
                if distance < candidate['distance']:
                    candidate['distance'] = distance
                    candidate['label'] = self.training_histograms_labels[index]

            validation.append(candidate['label'])

        print("[*] Validation accuracy: {}%".format(
            self.accuracy(validation, self.validation.labels) * 100)
        )
