# code modified from https://github.com/jacobkoenig/clDice-Loss
# algorithm credit: https://arxiv.org/abs/2003.07311
# @misc{shit2020cldice,
#     title={clDice -- a Topology-Preserving Loss Function for Tubular Structure Segmentation},
#     author={Suprosanna Shit and Johannes C. Paetzold and Anjany Sekuboyina and Andrey Zhylka and Ivan Ezhov and Alexander Unger and Josien P. W. Pluim and Giles Tetteh and Bjoern H. Menze},
#     year={2020},
#     eprint={2003.07311},
#     archivePrefix={arXiv},
#     primaryClass={cs.CV}
# }

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K


def _move_channel_to_first(*args):
    results = list()
    for x in args:
        rank = x.shape.rank
        if rank == 4:
            results.append(tf.transpose(x, (0, 3, 1, 2)))
        elif rank == 3:
            results.append(tf.transpose(x, (2, 0, 1)))
        elif rank == 2:
            results.append(x)
        else:
            results.append(x)
            print(f'expects no more than 4D tensor but got shape of {x.shape}')

    if len(results) == 1:
        return results[0]
    else:
        return (results)


def _move_channel_to_last(*args):
    results = list()
    for x in args:
        rank = x.shape.rank
        if rank == 4:
            results.append(tf.transpose(x, (0, 2, 3, 1)))
        elif rank == 3:
            results.append(tf.transpose(x, (1, 2, 0)))
        elif rank == 2:
            results.append(x)
        else:
            results.append(x)
            print(f'expects no more than 4D tensor but got shape of {x.shape}')

    if len(results) == 1:
        return results[0]
    else:
        return (results)


def _cast_to(*args, dtype=tf.float32):
    results = list()
    for x in args:
        results.append(tf.cast(x, dtype))
    
    if len(results) == 1:
        return results[0]
    else:
        return (results)


def _maybe_trim_logits(pred, target, data_format="channels_first"):
    if data_format == 'channels_first':
        pred, target = _move_channel_to_last(pred, target)

    if (target.shape[-1] == 1) and (pred.shape[-1] == 2):
        # other_sizes = pred.shape[:-1]
        pred = tf.gather(pred, [-1], axis=-1)
        # pred = tf.reshape(pred, [*other_sizes, 1])

    return pred, target


def dice_loss(data_format="channel_last"):
    """dice loss function for tensorflow/keras
        calculate dice loss per batch and channel of each sample.
    Args:
        data_format: either channels_first or channels_last
    Returns:
        loss_function(y_true, y_pred)  
    """

    def loss(target, pred):
        pred, target = _maybe_trim_logits(pred, target, data_format)
        pred, target = _move_channel_to_first(pred, target)
        pred, target = _cast_to(pred, target, dtype=tf.float32)
        
        smooth = 1.0
        iflat = tf.reshape(
            pred, (tf.shape(pred)[0], tf.shape(pred)[1], -1)
        )  # batch, channel, -1
        tflat = tf.reshape(target, (tf.shape(target)[0], tf.shape(target)[1], -1))
        intersection = K.sum(iflat * tflat, axis=-1)
        return 1 - ((2.0 * intersection + smooth)) / (
            K.sum(iflat, axis=-1) + K.sum(tflat, axis=-1) + smooth
        )

    return loss


def soft_skeletonize(x, thresh_width=10):
    """
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - needs to be greater then or equal to the maximum radius for the tube-like structure
    """

    minpool = (
        lambda y: K.pool2d(
            y * -1,
            pool_size=(3, 3),
            strides=(1, 1),
            pool_mode="max",
            data_format="channels_first",
            padding="same",
        )
        * -1
    )
    maxpool = lambda y: K.pool2d(
        y,
        pool_size=(3, 3),
        strides=(1, 1),
        pool_mode="max",
        data_format="channels_first",
        padding="same",
    )

    for i in range(thresh_width):
        min_pool_x = minpool(x)
        contour = K.relu(maxpool(min_pool_x) - min_pool_x)
        x = K.relu(x - contour)
    return x


def norm_intersection(center_line, vessel):
    """
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    """
    smooth = 1.0
    clf = tf.reshape(
        center_line, (tf.shape(center_line)[0], tf.shape(center_line)[1], -1)
    )
    vf = tf.reshape(vessel, (tf.shape(vessel)[0], tf.shape(vessel)[1], -1))
    intersection = K.sum(clf * vf, axis=-1)
    return (intersection + smooth) / (K.sum(clf, axis=-1) + smooth)


class SoftClDice:
    """clDice loss function for tensorflow/keras
    Args:
        k: needs to be greater or equal to the maximum radius of the tube structure.
        data_format: either channels_first or channels_last        
    Returns:
        loss_function(y_true, y_pred)  
    """
    def __init__(self, k=10, data_format="channels_last"):
        self.k = k
        self.data_format = data_format

    def loss(self, target, pred, data_format=None):
        if data_format is None:
            data_format = self.data_format
        pred, target = _maybe_trim_logits(pred, target, data_format)
        pred, target = _move_channel_to_first(pred, target)
        pred, target = _cast_to(pred, target, dtype=tf.float32)

        cl_pred = soft_skeletonize(pred, thresh_width=self.k)
        target_skeleton = soft_skeletonize(target, thresh_width=self.k)
        iflat = norm_intersection(cl_pred, target)
        tflat = norm_intersection(target_skeleton, pred)
        intersection = iflat * tflat
        return tf.reduce_mean(1 - ((2.0 * intersection) / (iflat + tflat)))
