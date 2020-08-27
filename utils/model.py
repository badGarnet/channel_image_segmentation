#! /usr/bin/env python

import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa


class ChannelCutter:
    def __init__(self, preprocessing=True):
        self.preprocessing = preprocessing

    @staticmethod
    def preprocess(images, labels):
        batch_size, i_label = images.shape[[0, -1]]
        batch_data = tf.concat([images, labels])
        angles = np.pi * np.random.randint(0, 360, batch_size) / 180
        rotated = tfa.image.rotate(batch_data, angles=angles)
        return batch_data[:, :, :, :i_label], batch_data[:, :, :, i_label]

    @staticmethod
    def load_multi_channel_data(path, extension, features, masks):
        fnames = [f for f in os.listdir(path) if f.endswith(extension)]
        # defin loader by file extension
        # TODO: enable .mat file
        loader = {
            '.npy': np.load,
        }.get(extension)
        combiner = {
            '.npy': np.concatenate
        }.get(extension)

        data = list()
        for f in fnames:
            data.append(loader(path / f))

        data = combiner(data)
        return data[:, :, :, features], data[:, :, :, masks]
