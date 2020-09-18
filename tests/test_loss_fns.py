import unittest
from utils.cldice_loss import *
from utils.cldice_loss import _move_channel_to_first, _maybe_trim_logits, _cast_to
from utils.focal_tversky_loss import mean_iou_loss, reduced_iou_loss, weighted_ce_loss
import tensorflow as tf


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.target4 = tf.random.uniform(shape=(10, 300, 200, 1), maxval=2, dtype=tf.dtypes.int32)
        self.target3 = tf.random.uniform(shape=(300, 200, 1), maxval=2, dtype=tf.dtypes.int32)
        self.bad_pred_4 = 1 - self.target4
        self.bad_pred_3 = 1 - self.target3
        self.pred_4 = tf.random.uniform(shape=(10, 300, 200, 2), maxval=1, dtype=tf.dtypes.float32)
        self.target3 = tf.random.uniform(shape=(300, 200, 2), maxval=1, dtype=tf.dtypes.float32)
        self.funcs = {
            'cldice': SoftClDice().loss,
            'iou': mean_iou_loss,
            'reduced_iou': reduced_iou_loss,
            'weighted_ce': weighted_ce_loss
        }

    def test_move_channel_to_first(self):
        pred, target = _move_channel_to_first(self.pred_4, self.target4)
        self.assertEqual(2, pred.shape[1])
        self.assertEqual(1, target.shape[1])

    def test_move_channel_to_first_one(self):
        pred = _move_channel_to_first(self.pred_4)
        self.assertEqual(2, pred.shape[1])

    def test_maybe_trim_logits(self):
        pred, target = _maybe_trim_logits(self.pred_4, self.target4, data_format='channels_last')
        self.assertEqual(1, pred.shape[-1])
        self.assertEqual(1, target.shape[-1])

    # TODO: define success per loss function
    def test_perfect_match4(self):
        for key, func in self.funcs.items():
            with self.subTest(func=func):
                self.assertEqual(0, int(func(self.target4, self.target4)))

    def test_perfect_mismatch4(self):
        for key, func in self.funcs.items():
            with self.subTest(func=func):
                self.assertEqual(1, int(func(self.target4, self.bad_pred_4)))