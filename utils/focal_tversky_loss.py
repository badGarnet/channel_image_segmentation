# credit: https://github.com/nabsabraham/focal-tversky-unet
# citation:
# @article{focal-unet,
#   title={A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation},
#   author={Abraham, Nabila and Khan, Naimul Mefraz},
#   journal={arXiv preprint arXiv:1810.07842},
#   year={2018}
# }

from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf 

epsilon = 1e-5
smooth = 1

def prepare(y_true, y_pred):
    if y_pred.shape[-1] > y_true.shape[-1]:
        y_pred_probs = tf.gather(y_pred, indices=1, axis=-1)
    else:
        y_pred_probs = tf.gather(y_pred, indices=0, axis=-1)

    y_true_probs = tf.gather(y_true, indices=0, axis=-1)
    
    return tf.cast(y_true_probs, tf.float32), tf.cast(y_pred_probs, tf.float32)

def dsc(y_true_in, y_pred_in):
    y_true, y_pred = prepare(y_true_in, y_pred_in)
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = 2 * (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true_in, y_pred_in):
    y_true, y_pred = prepare(y_true_in, y_pred_in)
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def confusion(y_true_in, y_pred_in):
    y_true, y_pred = prepare(y_true_in, y_pred_in)
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true_in, y_pred_in):
    y_true, y_pred = prepare(y_true_in, y_pred_in)
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true_in, y_pred_in):
    y_true, y_pred = prepare(y_true_in, y_pred_in)
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def tversky(y_true_in, y_pred_in):
    y_true, y_pred = prepare(y_true_in, y_pred_in)
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


def generalized_dice_loss(y_true_in, y_pred_in, n_class=2, sigma=1.):
    if len(y_true_in.shape) == 3:
        onehot = tf.one_hot(tf.gather(y_true_in, indices=0, axis=-1), depth=n_class)
    else:
        onehot = tf.one_hot(y_pred_in, depth=n_class)

    generalised_dice_numerator = 0.
    generalised_dice_denominator = 0.
    for i in range(onehot.shape[-1]):
        y_true = tf.gather(onehot, indices=i, axis=-1)
        y_pred = tf.gather(y_pred_in, indices=i, axis=-1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        ref_vol = tf.keras.backend.flatten(y_true)
        intersect = tf.keras.backend.flatten(y_true * y_pred)
        seg_vol = tf.keras.backend.flatten(y_pred)
        weight = 1. / tf.maximum(sigma, tf.reduce_sum(ref_vol))
        generalised_dice_numerator += 2. * weight * tf.reduce_sum(intersect)
        generalised_dice_denominator += weight * tf.reduce_sum(seg_vol + ref_vol)

    generalised_dice_score = \
        generalised_dice_numerator / (generalised_dice_denominator + sigma)
    return 1 - generalised_dice_score