#! /usr/bin/env python

import numpy as np
import math
import random
from numpy.core.shape_base import block
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, \
    BatchNormalization 


class ChannelCutter:
    def __init__(self, config):
        self._config = config
        self._base_layer_names = config.get('base_layer_names', list())
        self._base_config = config.get('base_config', None)
        self._input_shape = config.get('input_shape', None)
        self.n_classes = config.get('n_classes', None)
        self._base_layers = None
        self.base_model = None
        self._compiled = False

        if self._base_config is not None:
            self.base_model = config.get(
                'base_model', tf.keras.applications.MobileNetV2
            )
            self.base_model = self.base_model(**self._base_config)

            if len(self._base_layer_names) > 0:
                self._base_layers = self._get_base_layers(self._base_layer_names)
                self.unet = None
        else:
            self.unet = self._build_simple_unet(self._input_shape, self.n_classes)

    def compile_model(self, optimizer, loss, **kwargs):
        self.unet.compile(optimizer=optimizer, loss=loss, **kwargs)
        self._compiled = True
        return self.unet

    def fit(self, X, y):
        if not self._compiled:
            raise ValueError("model is not compiled yet")
        return self.unet.fit(X, y)
        
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
                n_filters, kernel_size, padding='same',
                name=f'down_conv_block{i_block}_step{i}', 
                **kwargs
            )(x)

        pre_pool = x
        x = MaxPool2D(pool_size=(pool_size), name=f'maxpool_block{i_block}')(x)
        return x, pre_pool

    @staticmethod
    def up_block(i_block, pre_pool, x, pool_size, n_filters, n_steps, kernel_size, **kwargs):
        x = Conv2DTranspose(
            filters=x.shape[-1], kernel_size=kernel_size, strides=pool_size, 
            padding='same', name=f'upsample_block{i_block}'
        )(x)
        x = tf.concat([x, pre_pool], axis=-1)
        for i in range(n_steps):
            x = Conv2D(
                n_filters, kernel_size, padding='same',
                name=f'up_conv_block{i_block}_step{i}', 
                **kwargs
            )(x)
        return x

    @staticmethod
    def basic_block(x, n_filters, kernel_size, batch_norm=True, **kwargs):
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', **kwargs)(x)
        if batch_norm:
            x = BatchNormalization()(x)

        return x

    @staticmethod
    def _build_simple_unet(
        input_shape, n_classes=2, activation='relu',
        down_steps=3, init_kernels=64, init_kernel_size=7, 
        other_kernel_size=3, pool_size=2, block_n_steps=3,
        pre_unet_conv_steps=2, batch_norm=True):
        inp = tf.keras.Input(shape=(input_shape))
        x = ChannelCutter.basic_block(
            inp, n_filters=init_kernels, kernel_size=init_kernel_size, 
            batch_norm=batch_norm, name=f'input_conv'
        )

        for i_pre in range(pre_unet_conv_steps):
            x = ChannelCutter.basic_block(
                x, n_filters=init_kernels, kernel_size=other_kernel_size, 
                batch_norm=batch_norm, name=f'pre_unet_conv_step{i_pre}',
                activation=activation
            )

        n_filters = [init_kernels]
        skip_layers = list()
        for i_down in range(down_steps):
            new_filters = n_filters[-1] * 2
            x, skip_layer = ChannelCutter.down_block(
                i_down, x, n_filters=new_filters, kernel_size=other_kernel_size, 
                n_steps=block_n_steps, pool_size=pool_size, activation=activation,
            )
            n_filters.append(new_filters)
            skip_layers.append(skip_layer)

        for i in range(block_n_steps):
            x = Conv2D(
                filters=n_filters[-1] * 2, kernel_size=other_kernel_size, activation=activation,
                padding='same', name=f'up_conv_u_step{i}', 
            )(x)

        for i_up in range(down_steps):
            x = ChannelCutter.up_block(
                i_up, skip_layers[-i_up-1], x, pool_size=pool_size,
                n_filters=n_filters[-i_up-1], n_steps=block_n_steps,
                kernel_size=other_kernel_size, activation=activation
            )

        x = Conv2D(filters=n_classes, kernel_size=1, activation=activation)(x)
        x = tf.keras.activations.softmax(x)

        return tf.keras.Model(inputs=inp, outputs=x)
