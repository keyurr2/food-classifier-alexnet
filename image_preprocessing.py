#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:13:22 2018

@author: keyur-r
"""

import os
import shutil

# base_path = "dataset"
# meta_path = "dataset/meta"
# source_path = "dataset/images"
# meta - train.txt, input_dir - file to move from


class ImageProcessor():

    def __init__(self, base_path, meta_path, source_path, dataset):
        self.base_path = base_path
        self.meta_path = meta_path
        self.source_path = source_path
        self.dataset = dataset

    def create_folder(self):
        self.dest_path = os.path.join(self.base_path, self.dataset)
        os.makedirs(self.dest_path, exist_ok=True)

    def create_subfolder(self):
        with open(self.meta_path + "/classes.txt") as f:
            for line in f:
                os.makedirs(os.path.join(self.dest_path, line.rstrip('\n')), exist_ok=True)

    def prepare_dataset(self):
        meta_file = os.path.join(self.meta_path, self.dataset + ".txt")
        with open(meta_file) as f:
            for line in f:
                category, file = line.strip('\n').split("/")
                source_file = os.path.join(
                    self.source_path, category, str(file) + ".jpg")
                dest_file = os.path.join(
                    self.dest_path, category, str(file) + ".jpg")
                shutil.copy(source_file, dest_file)
#                print(source_file, dest_file)

img_processor = ImageProcessor(
    "dataset", "dataset/meta", "dataset/images", "train")

img_processor.create_folder()
img_processor.create_subfolder()
img_processor.prepare_dataset()
