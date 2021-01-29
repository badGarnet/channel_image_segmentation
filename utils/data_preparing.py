import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
from inspect import getfullargspec
import math
import random
import shutil


def move_files_into_train_test(
    path, base_pattern, seed=42, splits={'train': 0.7, 'test': 0.3}):
    files = os.listdir(path)
    index = [f for f in files if base_pattern in f]
    total_sample_size = len(index)
    np.random.seed(seed)
    np.random.shuffle(index)
    split_index = {}
    for key, val in splits.items():
        end = int(total_sample_size * val)
        split_index[key] = index[:end]
        index = index[end:]
    
    if len(index) > 1:
        split_index[key].extend(index)

    path = Path(path)
    for key, indices in split_index.items():
        save_path = path / key
        os.makedirs(save_path, exist_ok=True)
        for idx in indices:
            to_move = [f for f in files if idx.split(base_pattern)[0] in f]
            for f in to_move:
                shutil.move(str(path / f), str(save_path))
                files.remove(f)


def get_angles(tensor):
    if len(tensor.shape) > 3:
        angles = tf.random.uniform(
            shape=[tensor.shape[0]],
            minval=-10. * math.pi / 180.,
            maxval=10. * math.pi / 180.,
        )
    else:
        angles = tf.random.uniform(
            shape=[1],
            minval=-10. * math.pi / 180.,
            maxval=10. * math.pi / 180.,
        )
    # else:
    #     random.seed(seed)
    #     angles = random.randint(0, 360) * math.pi / 180

    return angles


def random_rotation(tensor):
    angles = get_angles(tensor)
    return tfa.image.rotate(tensor, angles)


def maybe_pad_image(x, input_shape=None):
    if input_shape is None:
        input_shape = x.shape

    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    height, width = input_shape[0], input_shape[1]
    # padding input to be a square image if necessary
    pad = ~(height == width)
    if pad:
        if height > width:
            h_offset = 0
            w_offset = int((height - width) / 2)
            target = height
        else:
            w_offset = 0
            h_offset = int((width - height) / 2)
            target = width
        return tf.image.pad_to_bounding_box(
            x, h_offset, w_offset, target, target
        )
    else:
        return x


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


def crop_image(image, img_height=512, img_width=320, img_channel=5):
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
        path / (prefix + 'image.png'), normalize_with_moments(tensor[:, :, :3])
    )
    elevation = tensor[:, :, 4:]
    tf.keras.preprocessing.image.save_img(
        path / (prefix + 'elevation.png'),  elevation
    )
    tf.keras.preprocessing.image.save_img(
        path / (prefix + 'mask.png'), tensor[:, :, 3:4] / 255
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
        self.feature_channels = 3 + len(kwargs)
        self._data_mapping()

    def _data_mapping(self, process=None):
        if process is None:
            for proc in self.process:
                self.data = self.data.map(proc)
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
    def process_path(path, mask_key, image_key, label_scaler=255, **kwargs):
        label = ChannelData.load_image(path) / tf.cast(label_scaler, tf.uint8)
        label = tf.cast(label, tf.uint8)
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
            n_process = self._bundled_process(process, feature_channels=self.feature_channels, **kwargs)
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
    def _bundled_process(process, feature_channels=4, **kwargs):

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
                i_features, i_label = tf.range(feature_channels), tf.range(feature_channels, feature_channels+1)

            # bundle data
            bundled = tf.concat([features, labels], axis=cat_axis)
            # process together
            processed = process(bundled)
            # recover feature and label
            new_f = tf.gather(processed, i_features, axis=cat_axis)
            new_l = tf.gather(processed, i_label, axis=cat_axis)
            return new_f, new_l

        return wrapper



def main():
    data_path = Path('./data')
    crop_path = data_path / 'crop_arkansas_512x320_moment_norm'
    os.makedirs(crop_path, exist_ok=True)
    # load all data into numpy
    fnames = [f for f in os.listdir(data_path) if (f.startswith('tbd_19') & f.endswith('.npy'))]

    for part, f in enumerate(fnames):
        data = np.load(data_path / f).astype(int)
        # data = load_multi_channel_data(data_path, '.npy')
        print(f'loaded dataset with the shape {data.shape}')
        # covert into a Dataset object to use batch processing (can't process all images on a PC)
        dslice = tf.data.Dataset.from_tensor_slices(data)
        # apply cropping: we run multiple passes
        n_pass = 2
        np.random.seed(42)
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
                    save_crops(
                        image, prefix=f'part_{part}_pass_{i_pass}_', batch=i, idx=j, 
                        path=crop_path)

    # move files into train, val, and test
    move_files_into_train_test(crop_path, 'mask', splits={
        'train': 0.6, 'val': 0.2, 'test': 0.2
    })

                    
def move_files():
    data_path = Path('./data') / 'crop_512x320'
    move_files_into_train_test(data_path, 'mask')


if __name__ == "__main__":
    main()
