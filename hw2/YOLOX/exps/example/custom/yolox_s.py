#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        # self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = "yolox_s_SE"

        # Define yourself dataset path
        self.data_dir = "datasets/dataset"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"

        self.num_classes = 1

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1

        self.basic_lr_per_img = 0.01 / 64.0

        self.save_history_ckpt = False

