import unittest
from utils.model import ChannelCutter
from utils.data_preparing import ChannelData, random_rotation
import tensorflow as tf
from pathlib import Path


class TestIntegrationModelPipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ChannelData(
            path=Path('./data/test_data')
        )
        cls.data.add_process(random_rotation)
        cls.model = ChannelCutter({
            'input_shape': [500, 300, 4],
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
