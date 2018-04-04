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

from mlp import MultiLayerPerceptron
from dataset import DataSet

if __name__ == "__main__":
    mlp = MultiLayerPerceptron()
    mlp.addLayer(2048)
    mlp.addLayer(2048)
    mlp.addLayer(300)
    mlp.setInputs(DataSet("datasets/Train/trainVectors.csv", "datasets/Train/trainLbls.csv"), DataSet("datasets/Validate/valVectors.csv", "datasets/Validate/valLbls.csv"))
    mlp.train()
