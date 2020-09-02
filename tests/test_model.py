import unittest
from utils.model import ChannelCutter
import tensorflow as tf


class TestChannelCutter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            'base_config': {
                'input_shape': [128, 128, 3],
                'include_top': False
            }
        }
        cls.basic_config = {
            'input_shape': [128, 128, 4],
            'n_classes': 2
        }

    def test_init(self):
        # generate from an existing application
        model = ChannelCutter(self.config)
        # should be able to get the following layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        for layer in layer_names:
            with self.subTest(layer=layer):
                self.assertTrue(
                    model.base_model.get_layer(layer) is not None
                )

    def test_init_with_missing_layers(self):
        # generate from an existing application
        model = ChannelCutter(self.config)
        # should be able to get the following layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'foo',      # 4x4
        ]
        actual_layers = model._get_base_layers(layer_names)
        for name, layer in zip(layer_names, actual_layers):
            with self.subTest(name=name):
                if name == 'foo':
                    self.assertTrue(layer is None)
                else:
                    self.assertEqual(name, layer.name)

    def test_unet_arch(self):
        model = ChannelCutter(self.basic_config)
        self.assertTrue(model.unet is not None)

    def test_compile(self):
        model = ChannelCutter(self.basic_config)
        model.compile_model(
            optimizer='adam', loss='mse', metrics=['accuracy']
        )
        self.assertTrue(
            tf.keras.utils.plot_model(model.unet, show_shapes=True)
        )
