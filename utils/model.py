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
    def __init__(self, config):
        self._config = config
        self._base_layer_names = config.get('base_layer_names', list())
        self._base_config = config.get('base_config', None)
        self.base_model = config.get(
            'base_model', tf.keras.applications.MobileNetV2
        )
        if self._base_config is not None:
            self.base_model = self.base_model(**self._base_config)
        
    def _get_base_layers(self):
        layers = list()
        for name in self._base_layer_names:
            try:
                layers.append(self.base_model.get_layer(name))
            except ValueError:
                print(f"can't get layer {name} from base model {self.base_model}")
                layers.append(None)
        return layers

