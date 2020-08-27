import unittest
from utils.model import ChannelCutter
from utils.data_preparing import ChannelData
import tensorflow as tf
from pathlib import Path
import os
from functools import partial


class TestDataPipe(unittest.TestCase):

    def setUp(self):
        self.test_tensor = tf.random.uniform(
            shape=(10, 200, 300, 5), minval=0, maxval=1
        )
        self.test_path=Path('./data/test_data')

    def test_get_angles_batch(self):
        angles = ChannelCutter().get_angles(self.test_tensor)
        self.assertEqual(
            self.test_tensor.shape[0], angles.shape[0]
        )

    def test_get_angles_oneof(self):
        angles = ChannelCutter().get_angles(self.test_tensor[0])
        self.assertTrue(
            isinstance(angles, float)
        )

    def test_init_data(self):
        cdata = ChannelData(path=self.test_path, elevation='elevation')
        self.assertEqual('mask', cdata.mask_key)

    def test_get_data_list(self):
        cdata = ChannelData(path=self.test_path, elevation='elevation')
        dlist = list(cdata.get_data_list().as_numpy_iterator())
        expected = [f for f in os.listdir(self.test_path) if 'mask' in f]
        actual = [str(f).split('/')[-1][:-1] for f in dlist]
        self.assertListEqual(sorted(expected), sorted(actual))

    def test_load_image(self):
        cdata = ChannelData(path=self.test_path, elevation='elevation')
        dlist = cdata.get_data_list()
        images = dlist.map(ChannelData.load_image)
        self.assertEqual(
            len(dlist), len(images)
        )

    def test_process_path(self):
        cdata = ChannelData(path=self.test_path, elevation='elevation')
        dlist = cdata.get_data_list()
        process = partial(
            ChannelData.process_path, 
            mask_key=cdata.mask_key,
            image_key=cdata.image_key,
            **cdata.others
        )
        labeled_ds = dlist.map(process)
        expected = [f for f in os.listdir(self.test_path) if 'mask' in f]
        self.assertEqual(
            len(expected), len(labeled_ds)
        )
        for image, label in labeled_ds.take(1):
            self.assertEqual(image.shape[-1], 4)
            self.assertEqual(image.shape[0], label.shape[0])