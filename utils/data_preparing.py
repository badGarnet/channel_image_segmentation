import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa


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
