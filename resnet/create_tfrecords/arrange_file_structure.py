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
new_dataset_path = sys.argv[2]

if not os.path.isdir(new_dataset_path):
    os.makedirs(new_dataset_path)

training_labels = open(training_set_path + "/trainLbls.csv", "r")

for index, label in enumerate(training_labels):
    label_path = new_dataset_path + "/" + label

    if not os.path.isdir(label_path):
        os.makedirs(label_path)

    shutil.copyfile(
        training_set_path + "/TrainImages/Image{}.jpg".format(index + 1),
        label_path + "/Image{}.jpg".format(index + 1)
    )

