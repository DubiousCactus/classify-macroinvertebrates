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
import tensorflow as tf

from mlp import MultiLayerPerceptron
from dataset import DataSet

if __name__ == "__main__":
    mlp = MultiLayerPerceptron()
    mlp.epochs = 30
    mlp.addLayer(2048)
    mlp.addLayer(2048)
    mlp.addLayer(128)
    mlp.setInputs(
        DataSet("datasets/Train/trainVectors.csv", "datasets/Train/trainLbls.csv"),
        DataSet("datasets/Validate/valVectors.csv", "datasets/Validate/valLbls.csv"),
        DataSet("datasets/Test/testVectors.csv")
    )

    if os.path.isfile("model.ckpt.index"):
        usr_input = 'n'
        usr_input = input("Restore model? [y/N]: ")

        if usr_input.lower() == 'y':
            # Restore variables from disk.
            saver = tf.train.Saver()
            sess = tf.Session()
            saver.restore(sess, "model.ckpt")
            print("Model restored.")
            # mlp.test(sess)
        else:
            mlp.train()
            # mlp.test()
    else:
        mlp.train()
        # mlp.test()
