import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
from inspect import getfullargspec


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


def crop_image(image, img_height=500, img_width=300, img_channel=5):
    """cropping image using tfa's random_crop method

    Args:
        image (Tensor): of shape (n_batch, height, width, img_channel)
        img_height (int, optional): height of crop area in pixels. Defaults to 500.
        img_width (int, optional): width of crop area in pixels. Defaults to 300.
        img_channel (int, optional): number of channels in the image. Defaults to 5.

    Returns:
        Tensor: cropped Tensor of shape (n_batch, img_height, img_width, img_channel)
    """
    cropped = tf.image.random_crop(
        image, [img_height, img_width, img_channel], 
        seed=None, name=None
    )
    return cropped


def save_crops(tensor, prefix='', batch=0, idx=0, path=Path('.')):
    """save tensor into images of RGB, elevation (4th channel) and mask (5th channel)
    make sure the input images are configured in such a way

    TODO: make this more flexible for more types of channels (e.g., slope)

    Args:
        tensor (Tensor): must have last dim's size equal to 5
        prefix (str, optional): prefix of the file. Defaults to ''.
        batch (int, optional): batch number. Defaults to 0.
        idx (int, optional): image number. Defaults to 0.
        path ([type], optional): path to save the files. Defaults to Path('.').
    """
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


class ChannelData:
    def __init__(self, path, mask_key='mask', image_key='image', **kwargs):
        self.path = path
        self.mask_key = mask_key
        self.image_key = image_key
        self.others = kwargs
        self.data = self.get_data_list()
        self.process = [partial(
            self.process_path,
            mask_key=mask_key,
            image_key=image_key,
            **kwargs
        )]
        self._data_mapping()

    def _data_mapping(self, process=None):
        if process is None:
            def all_processes(tensor):
                for proc in self.process:
                    tensor = proc(tensor)
                return tensor
            self.data = self.data.map(all_processes)
        else:
            self.data = self.data.map(process)

    def get_data_list(self):
        return tf.data.Dataset.list_files(
            str(self.path/('*'+self.mask_key+'*.png'))
        )

    @staticmethod
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img)
        return img

    @staticmethod
    def process_path(path, mask_key, image_key, **kwargs):
        label = ChannelData.load_image(path)
        image = [ChannelData.load_image(
            tf.strings.regex_replace(path, mask_key, image_key)
        )]
        for val in kwargs.values():
            image.append(
                ChannelData.load_image(tf.strings.regex_replace(
                    path, mask_key, val
                ))
            )

        return tf.concat(image, axis=-1), label

    def add_process(self, process, **kwargs):
        """adding a new process __after__ images are loaded and __before__ training

        Args:
            process (function): processing method; must take two arguments ``(features, labels)`` and returns
                transfomred ``(features, labels)`` pairs
        """
        args = getfullargspec(process)

        n_args = len(args.args)
        if n_args <= 1:
            n_process = self._bundled_process(process)
        elif n_args == 2:
            if len(kwargs) > 0:
                n_process = partial(process, **kwargs)
            else:
                n_process = process
        else:
            raise ValueError(f'process should take either one or two positional arguments but got {n_args}')
        self.process.append(n_process)
        self._data_mapping(n_process)
        return self

    @staticmethod
    def _bundled_process(process, **kwargs):

        @tf.function
        def wrapper(features, labels, cat_axis=-1):
            # get number of channels in feature tensor
            n_channels_feature = features.shape[cat_axis]
            # generate indices to gather for features and label
            if n_channels_feature is not None:
                i_features = tf.range(n_channels_feature)
                i_label = tf.range(n_channels_feature, n_channels_feature + labels.shape[cat_axis])
            else:
                # TODO: how to get n channels when all is (none, none, none)
                i_features, i_label = tf.range(4), tf.range(4, 5)

            # bundle data
            bundled = tf.concat([features, labels], axis=cat_axis)
            # process together
            processed = process(bundled, **kwargs)
            # recover feature and label
            new_f = tf.gather(processed, i_features, axis=cat_axis)
            new_l = tf.gather(processed, i_label, axis=cat_axis)
            return new_f, new_l

        return wrapper


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
            print(f'cropping pass {i_pass+1} of dataset {part}...', end='')
            # map cropping method onto the dataset
            cropped_slices = (
            dslice
                .shuffle(seeds[i_pass])
                .map(crop_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(10)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
            print('saving data')
            for i, batch in enumerate(cropped_slices):
                for j, image in enumerate(batch):
                    save_crops(image, prefix=f'part_{part}_', batch=i, idx=j, path=data_path / 'crops')


if __name__ == "__main__":
    main()
