#! /usr/bin/env python

import numpy as np
import math
import random
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa


class ChannelCutter:
    def __init__(self, preprocessing=True, dataset=None):
        self.preprocessing = preprocessing
        self.dataset = dataset

    @staticmethod
    def get_angles(tensor, master_seed=42):
        if len(tensor.shape) > 3:
            angles = tf.random.uniform(
                shape=[tensor.shape[0]],
                minval=0,
                maxval=2 * math.pi,
                seed=master_seed
            )
        else:
            angles = random.randint(0, 360) * math.pi / 180

        return angles
