#! /usr/bin/env python

import numpy as np
import math
import random
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, BatchNormalization


class ChannelCutter:
    def __init__(self, config):
        self._config = config
        self._base_layer_names = config.get('base_layer_names', list())
        self._base_config = config.get('base_config', None)
        self.base_model = config.get(
            'base_model', tf.keras.applications.MobileNetV2
        )
        self.layers = None

        if self._base_config is not None:
            self.base_model = self.base_model(**self._base_config)

        if len(self._base_layer_names) > 0:
            self._base_layers = self._get_base_layers(self._base_layer_names)
        
    def _get_base_layers(self, names=None):
        layers = list()
        if names is None:
            names = self._base_layer_names
        for name in names:
            try:
                layers.append(self.base_model.get_layer(name))
            except ValueError:
                print(f"can't get layer {name} from base model {self.base_model}")
                layers.append(None)
        return layers

    def _get_unet(self, layers=None):
        if layers is None:
            layers = self._base_layers

        down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=layers)

        down_stack.trainable = True


    @staticmethod
    def down_block(i_block, x, n_filters, n_steps, kernel_size=3, pool_size=2, **kwargs):
        for i in range(n_steps):
            x = Conv2D(
                n_filters, kernel_size, 
                name=f'down_conv_block{i_block}_step{i}', 
                **kwargs
            )(x)

        pre_pool = x
        x = MaxPool2D(size=(pool_size), name=f'maxpool_block{i_block}')(x)
        return x, pre_pool

    @staticmethod
    def up_block(i_block, pre_pool, x, pool_size, n_filters, n_steps, kernel_size, **kwargs):
        x = Conv2DTranspose(filters=x.shape[-1], kernel_size=pool_size, name=f'upsample_block{i_block}')(x)
        x = tf.concat([x, pre_pool], axis=-1)
        for i in range(n_steps):
            x = Conv2D(
                n_filters, kernel_size, 
                name=f'up_conv_block{i_block}_step{i}', 
                **kwargs
            )(x)
        return x

    @staticmethod
    def _build_simple_unet(
        input_shape, down_steps=4, init_kernels=64, init_kernel_size=7, 
        other_kernel_size=3, pool_size=2,
        pre_unet_conv_steps=2, batch_normalization=True):
        inp = tf.keras.Input(shape=(input_shape))
        x = Conv2D(filters=init_kernels, kernel_size=init_kernel_size)(inp)
        if batch_normalization:
            x = BatchNormalization()(x)
        return x
