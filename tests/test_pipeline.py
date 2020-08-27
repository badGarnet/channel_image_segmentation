import unittest
from utils.model import ChannelCutter
import tensorflow as tf


class TestDataPipe(unittest.TestCase):

    def setUp(self):
        self.test_tensor = tf.random.uniform(
            shape=(10, 200, 300, 5), minval=0, maxval=1
        )

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