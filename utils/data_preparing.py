import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa


def load_multi_channel_data(path, extension, features=None, masks=None, process=None):
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
        d = loader(path / f)
        if process is not None:
            process(d)
        data.append(d)

    data = combiner(data)
    if features is None:
        return data
    else:
        return data[:, :, :, features], data[:, :, :, masks]


def random_crop_duplicator(data, n_crops_per_image=2, batch=10):
    return data



def main():
    data_path = Path('./data')
    data = load_multi_channel_data(data_path, '.npy')
    print(data.shape)


if __name__ == "__main__":
    main()
