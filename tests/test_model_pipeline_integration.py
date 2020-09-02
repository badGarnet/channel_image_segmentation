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
            path=Path('./data/test_data'),
            elevation='elevation'
        )
        cls.data.add_process(random_rotation)
        cls.model = ChannelCutter({
            'input_shape': [512, 320, 4],
            'n_classes': 2
        })
        cls.model.compile_model(
            optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
        )

    def test_feed_data(self):
        BATCH_SIZE = 2
        BUFFER_SIZE = 6

        train_dataset =  self.data.data\
            .shuffle(BUFFER_SIZE)\
            .batch(BATCH_SIZE)\
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.model.unet.fit(
            train_dataset, epochs=2
        )

        result = self.model.unet.evaluate(train_dataset)
        print(dict(zip(self.model.unet.metrics_names, result)))
        self.assertTrue(result[0] > 0)
