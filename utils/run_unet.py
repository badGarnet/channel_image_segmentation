import tensorflow as  tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sys
from pathlib import Path
import shutil
import datetime
from functools import partial
import json

from data_preparing import ChannelData, random_rotation
from model import ChannelCutter, LRFinder, save_predictions
from focal_tversky_loss import *


# first run use fild_lr to find learning rate
find_lr = False

# ===============================================================
# setting up the data source folders and network hyper-parameters

# setup loss function with weight for false positive results
loss = IOULoss(fp_weight=0.25)
# used for model name tagging
session_labels = ['arkansas', 'crop512x320', 'no_rotation', 'fp025']

train_data = ChannelData(
    path=Path('../data/crop_arkansas_512x320/train'),
    elevation='elevation'
)
val_data = ChannelData(
    path=Path('../data/crop_arkansas_512x320/val'),
    elevation='elevation'
)
train_data.add_process(random_rotation)
val_data.add_process(random_rotation)

model = ChannelCutter({
    'input_shape': [512, 320, 4],
    'n_classes': 2
})

BATCH_SIZE = 2
BUFFER_SIZE = 6
EPOCHS = 2

train_dataset =  train_data.data\
    .shuffle(BUFFER_SIZE)\
    .batch(BATCH_SIZE)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset =  val_data.data\
    .shuffle(BUFFER_SIZE)\
    .batch(BATCH_SIZE)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# ===============================================================
# lr finding if find_lr = true
image_name = 'logs/lr_' + '_'.join(session_labels) + '.png'
if find_lr:
    model.compile_model(
        optimizer='adam', 
        loss=loss.loss, 
        metrics=['accuracy']
    )

    lr_finder = LRFinder(start_lr=1e-7, end_lr=10, max_steps=336)
    model.unet.fit(
        train_dataset, epochs=2, 
        callbacks=[lr_finder]
    )
    
    lr_finder.plot()
    plt.savefig(image_name)
    print(f'saved learning rate searching results to {image_name}')
    print(f'please open it and select the ideal learning rate for training a model')
    exit()
else:
    # use the find_lr section above to select a good learning rate and schedule (going from
    # initial_learning_rate to initial_learning_rate * decay_rate in decay_steps)
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.1,
        staircase=True)

    model.compile_model(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
        loss=reduced_iou_loss, 
        metrics=['accuracy', tp, tn]
    )

    # add learning rate to session labels
    session_labels.append('lr1e-3_to_lr1e-4')
    session_name = '_'.join(session_labels)
    save_path = Path("./logs") / session_name / ('trained_on_'+str(datetime.datetime.now()))
    os.makedirs(save_path)
    # callback that records training metrics for diagnosis
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_path)
    # callback that stops training if the monitored metric doesn't improve after specified epochs of training (patience)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # training the model
    model.unet.fit(
        train_dataset, epochs=EPOCHS, validation_data=val_dataset,
        callbacks=[tensorboard_callback, early_stopping_callback]
    )

    # save model artifacts
    with open(save_path / 'model_config.json', 'w') as fp:
        fp.write(str(model.unet.get_config()))

    model_path = str(save_path).replace('logs', 'models')
    os.makedirs(model_path)
    model.unet.save(model_path)
    print(f'model saved to {model_path}')

    # examine 5 sample validation images to visually check model performance beyond metrics
    print('generating predictions for training and validation datasets')
    save_predictions(model.unet, train_data.data, save_path / 'train_preds', prefix='train_')
    save_predictions(model.unet, val_data.data, save_path / 'val_preds', prefix='val_')
    # n_samples = 5
    # fig, axes = plt.subplots(nrows=3, ncols=n_samples, figsize=(n_samples*5, 15))
    # i = 0
    # tf.random.set_seed(2)
    # for x, y in val_dataset.take(5):
    #     axes[0][i].imshow(x[0, :, : :])
    #     axes[1][i].imshow(y[0, :, : :])
    #     axes[2][i].imshow(model.unet.predict(x)[0, :, :, 1])
    #     i += 1

    # plt.show()
    # plt.savefig(save_path / 'example_vals_1.png')
    # print(f'5 example results from the validation set are saved at {save_path}')

