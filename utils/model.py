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
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

from math import ceil
import PIL.Image as Image


def _mock_model(img):
    return img


def split_process_stitch_images(img, height, width, process=_mock_model, name='test'):
    """split an image of shape [height, width, channel] into segments, then 
    apply `process` to each segment, finally stitch the processed results
    back to [height, width, channel] shaped array"""
    img_h, img_w = img.shape[0], img.shape[1]
    assert img_h >= height
    assert img_w >= width
    
    n_h = ceil(img_h / height)
    n_w = ceil(img_w / width)
    
    splits = []
    corners = []
    for i_h in range(n_h):
        for i_w in range(n_w):
            if i_h < n_h - 1:
                bot, top = i_h * height, (i_h + 1) * height
            else:
                bot, top = img_h - height, img_h
                
            if i_w < n_w - 1:
                left, right = i_w * width, (i_w + 1) * width,
            else:
                left, right = img_w - width, img_w
                
            i_img = img[bot:top, left:right, :]
            
            splits.append(i_img)
            corners.append([bot, top, left, right])
            
    batch = np.array(splits)
    batch = np.moveaxis(batch, -1, 0)
    masks = process(batch)
    stitch = np.zeros(img.shape[:2])
    
    for mask, corner in zip(masks, corners):
        bot, top, left, right = corner
        stitch[bot:top, left:right] = mask
    
    return stitch


def save_predictions(model, dataset, save_path, prefix=''):
    """saving predicted masks with original image, DEM, and human interpreted mask

    Example:
        `save_predictions(train_data.data, Path('train_results'))`

    Args:
        model (tf.keras.Model): a trained model that has `predict(x)` method to make a prediction
        dataset (tf.data.Dataset): dataset to be used for prediction; must be tuples of (x, y) for each member; 
            must not be batched
        save_path (pathlib.Path): place to save the images
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    i = 0
    for x, y in dataset.batch(1):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        axes[0].imshow(x[0, :, :, :3])
        axes[1].imshow(x[0, :, :, 3])
        axes[2].imshow(y[0, :, : :])
        axes[3].imshow(model.predict(x)[0, :, :, 1])
        i += 1
        plt.savefig(save_path / f'{prefix}preds_{str(i)}.png')
        plt.close()


class LRFinder(Callback):
    """Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the plot method.
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 336, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)


class ChannelCutter:
    def __init__(self, config):
        self._config = config
        self._base_layer_names = config.get('base_layer_names', list())
        self._base_config = config.get('base_config', None)
        self._input_shape = config.get('input_shape', None)
        self.n_classes = config.get('n_classes', None)
        self.activation = config.get('activation', 'relu')
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
            self.unet = self._build_simple_unet(
                self._input_shape, self.n_classes,
                activation=self.activation
            )

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
    def down_block(i_block, x, n_filters, n_steps, kernel_size=3, pool_size=2, batch_norm=True, **kwargs):
        for i in range(n_steps):
            x = Conv2D(
                n_filters, kernel_size, padding='same',
                name=f'down_conv_block{i_block}_step{i}', 
                **kwargs
            )(x)
            if batch_norm:
                x = BatchNormalization()(x)

        pre_pool = x
        x = MaxPool2D(pool_size=(pool_size), name=f'maxpool_block{i_block}')(x)
        return x, pre_pool

    @staticmethod
    def up_block(i_block, pre_pool, x, pool_size, n_filters, n_steps, kernel_size, batch_norm=True, **kwargs):
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
            if batch_norm:
                x = BatchNormalization()(x)
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
        down_steps=3, init_kernels=32, init_kernel_size=7, 
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
            if batch_norm:
                x = BatchNormalization()(x)

        for i_up in range(down_steps):
            x = ChannelCutter.up_block(
                i_up, skip_layers[-i_up-1], x, pool_size=pool_size,
                n_filters=n_filters[-i_up-1], n_steps=block_n_steps,
                kernel_size=other_kernel_size, activation=activation
            )

        x = Conv2D(filters=n_classes, kernel_size=1, activation=activation)(x)
        x = tf.keras.activations.softmax(x)

        return tf.keras.Model(inputs=inp, outputs=x)
