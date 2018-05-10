#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Create the right file structure for the photos
"""

import os
import sys
import shutil

training_set_path = sys.argv[1]
validation_set_path = sys.argv[2]
new_training_dataset_path = sys.argv[3] + '/micro_invertebrae/micro_invertebrae_photos'
new_validation_dataset_path = sys.argv[4] + '/micro_invertebrae/micro_invertebrae_photos'

if not os.path.isdir(new_training_dataset_path):
    os.makedirs(new_training_dataset_path)

if not os.path.isdir(new_validation_dataset_path):
    os.makedirs(new_validation_dataset_path)

training_labels = open(training_set_path + "/trainLbls.csv", "r")
validation_labels = open(validation_set_path + "/valLbls.csv", "r")

for index, label in enumerate(training_labels):
    label = label.strip('\n')
    label_path = new_training_dataset_path + "/" + label

    if not os.path.isdir(label_path):
        os.makedirs(label_path)

    shutil.copyfile(
        training_set_path + "/TrainImages/Image{}.jpg".format(index + 1),
        label_path + "/Image{}.jpg".format(index + 1)
    )

for index, label in enumerate(validation_labels):
    label = label.strip('\n')
    label_path = new_validation_dataset_path + "/" + label

    if not os.path.isdir(label_path):
        os.makedirs(label_path)

    shutil.copyfile(
        validation_set_path + "/ValidationImages/Image{}.jpg".format(index + 1),
        label_path + "/Image{}.jpg".format(index + 1)
    )
