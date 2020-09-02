import unittest
from utils.model import ChannelCutter
from utils.data_preparing import ChannelData, random_rotation, maybe_pad_image
import tensorflow as tf
from pathlib import Path
from functools import partial


class TestIntegrationModelPipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ChannelData(
            path=Path('./data/test_data')
        )
        padding = partial(maybe_pad_image, input_shape=[500, 300, 4])
        cls.data.add_process(padding)
        cls.data.add_process(random_rotation)
        cls.model = ChannelCutter({
            'input_shape': [500, 500, 4],
            'n_classes': 2
        })
        cls.model.compile_model(
            optimizer='adam', loss=tf.keras.loss.BinaryCrossEntropy(), metrics=['accuracy']
        )

    def test_feed_data(self):
        train_dataset =  self.data.data.shuffle(1024).batch(4)
        self.model.unet.fit(train_dataset, epochs=2)
        result = self.model.unet.evaluate(train_dataset)
        print(dict(zip(self.model.unet.metrics_names, result)))
        self.assertTrue(result[0] > 0)
