import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa


def load_multi_channel_data(path, extension, features=None, masks=None, process=None):
    """loading data from a directory with a given extension, e.g., ``.npy``. Optionally process
    the data on a per file basis

    Args:
        path (Posix Path): path to the data files
        extension (str): current option is only ``.npy`` for numpy ndarray data
        features (list, optional): channels for the features, e.g., ``[0, 1, 2]``. Defaults to None.
        masks (list, optional): channels for the masks, e.g., ``[3]``. Defaults to None.
        process (function, optional): a process that takes in an ndarray. Defaults to None.

    Returns:
        numpy.ndarray: all data loaded into one numpy array; this can be large
    """
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
        else:
            data.append(d)

    data = combiner(data)
    if features is None:
        return data
    else:
        return data[:, :, :, features], data[:, :, :, masks]


def crop_image(image, idx=0, img_height=500, img_width=300, img_channel=5):
    cropped = tf.image.random_crop(
        image, [img_height, img_width, img_channel], 
        seed=None, name=None
    )
    return cropped


def save_crops(tensor, prefix='', batch=0, idx=0, path=Path('.')):
    prefix += f'batch_{batch}_num_{idx}_'
    tf.keras.preprocessing.image.save_img(
        path / (prefix + 'image.png'), tensor[:, :, :3]
    )
    tf.keras.preprocessing.image.save_img(
        path / (prefix + 'elevation.png'), tensor[:, :, 3:4]
    )
    tf.keras.preprocessing.image.save_img(
        path / (prefix + 'mask.png'), tensor[:, :, 4:]
    )


def main():
    data_path = Path('./data')
    # load all data into numpy
    fnames = [f for f in os.listdir(data_path) if f.endswith('.npy')]

    for part, f in enumerate(fnames):
        data = np.load(data_path / f).astype(int)
        # data = load_multi_channel_data(data_path, '.npy')
        print(f'loaded dataset with the shape {data.shape}')
        # covert into a Dataset object to use batch processing (can't process all images on a PC)
        dslice = tf.data.Dataset.from_tensor_slices(data)
        # apply cropping: we run multiple passes
        n_pass = 2
        seeds = np.random.randint(0, 1000, n_pass)
        for i_pass in range(n_pass):
            print(f'cropping pass {i_pass+1}')
            cropped_slices = (
            dslice
                .shuffle(seeds[i_pass])
                .map(crop_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(10)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
            print('saveing data')
            for i, batch in enumerate(cropped_slices):
                for j, image in enumerate(batch):
                    save_crops(image, prefix=f'part_{part}_', batch=i, idx=j, path=data_path / 'crops')


if __name__ == "__main__":
    main()
