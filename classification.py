#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Main class
"""

import os

from mlp import MultiLayerPerceptron
from svm import SupportVectorMachine
from sift_svm import SIFT_SupportVectorMachine
from dataset import DataSet

if __name__ == "__main__":
    # mlp = MultiLayerPerceptron()
#     mlp.epochs = 100
    # mlp.addLayer(2048)
    # mlp.addLayer(1024)
    # mlp.setInputs(
        # DataSet("datasets/Train/trainVectors.csv", "datasets/Train/trainLbls.csv", merge_with="datasets/Validate/valVectors.csv"),
        # # DataSet("datasets/Train/trainVectors.csv", "datasets/Train/trainLbls.csv"),
        # DataSet("datasets/Validate/valVectors.csv", "datasets/Validate/valLbls.csv"),
        # DataSet("datasets/Test/testVectors.csv")
    # )

    # if os.path.isfile("model.ckpt.index"):
        # usr_input = 'n'
        # usr_input = input("Restore model? [y/N]: ")

        # if usr_input.lower() == 'y':
            # # Restore variables from disk.
            # saver = tf.train.Saver()
            # sess = tf.Session()
            # saver.restore(sess, "model.ckpt")
            # print("Model restored.")
            # # mlp.test(sess)
        # else:
            # mlp.train()
            # # mlp.test()
    # else:
        # mlp.train()
        # mlp.test()

    # svm = SupportVectorMachine()
    # svm.setInputs(
        # DataSet("datasets/Train/trainVectors.csv", "datasets/Train/trainLbls.csv", merge_with="datasets/Validate/valVectors.csv"),
        # # DataSet("datasets/Train/trainVectors.csv", "datasets/Train/trainLbls.csv"),
        # DataSet("datasets/Validate/valVectors.csv", "datasets/Validate/valLbls.csv"),
        # DataSet("datasets/Test/testVectors.csv")
    # )
    # svm.train()
    # svm.test()

    SIFT_SVM = SIFT_SupportVectorMachine()
    SIFT_SVM.setInputs(
        DataSet(vectors_path=None, images_path="datasets/Train/TrainImages/", labels_path="datasets/Train/trainLbls.csv"),
        DataSet(vectors_path=None, images_path="datasets/Validate/ValidationImages", labels_path="datasets/Validate/valLbls.csv"),
        DataSet(vectors_path=None, images_path="datasets/Test/TestImages")
    )
    SIFT_SVM.train()
