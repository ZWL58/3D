import csv

import philsm_dataset
# import model_archive

import sklearn.metrics
import tensorflow as tf
import os
import datetime
import math

from sklearn import preprocessing
# from tensorflow.python.training.summary_io import SummaryWriter
from tqdm import tqdm,trange
from tensorflow import keras
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from dataset import *
from pa_repointnet import *
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from test import *

tf.random.set_seed(123)

# hyper_para
epochs = 100
learn_rate = 1e-6

# dataset
# currents,philsms = get_dataset() # test.py
currents,philsms = get_current_philsm() # dataset.py
data_size = len(currents)
train_dataset_size = int(data_size*0.8)
val_dataset_size = data_size-int(data_size*0.8)
dataset = tf.data.Dataset.from_tensor_slices((currents, philsms)).shuffle(len(currents))
train_dataset = dataset.take(train_dataset_size).batch(100)
val_dataset = dataset.skip(train_dataset_size).take(val_dataset_size).batch(val_dataset_size)

# create model
pa_repointnet = PA_RePointNet(current_dim=1000, philsm_dim=1, repointnet_pam_flag=True)
optimizer=tf.keras.optimizers.Adam(learn_rate)

def loss_func(y_true,y_pred):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.cast(y_pred, y_true.dtype)
    return tf.abs(y_true-y_pred)

# Start training
train_loss_epochs,val_loss_epochs = [],[]
for epoch in trange(epochs):
    loss_epoch = []
    for data_batch in train_dataset:
        current = data_batch[0]
        philsm = data_batch[1]
        with tf.GradientTape() as tape:
            pa_repointnet.trainable = True

            predicted_philsm = pa_repointnet(current)

            loss = loss_func(predicted_philsm,philsm)
            loss_epoch.append(tf.reduce_mean(loss))

            gradients = tape.gradient(loss, pa_repointnet.trainable_variables)
            optimizer.apply_gradients(zip(gradients, pa_repointnet.trainable_variables))
    train_loss_epochs.append(tf.reduce_mean(loss_epoch))
    # Verification
    for data_batch in val_dataset:
        current = data_batch[0]
        philsm = data_batch[1]
        pa_repointnet.trainable = False
        predicted_philsm = pa_repointnet.predict(current)

        loss = loss_func(predicted_philsm,philsm)
        val_loss_epochs.append(tf.reduce_mean(loss))

    plt.plot(train_loss_epochs,label='train')
    plt.plot(val_loss_epochs, label='val')
    plt.legend()
    plt.savefig(f'./predicted_results/loss.png', bbox_inches='tight')
    plt.cla()
    plt.close()

with open("./predicted_results/loss.csv",'w',newline='')as file:
    writer = csv.writer(file)
    head = ['Epoch', 'Train Loss', 'Val Loss']
    writer.writerow(head)
    for index, train_loss in enumerate(train_loss_epochs):
        data = [index, train_loss.numpy(), val_loss_epochs[index].numpy()]
        writer.writerow(data)

pa_repointnet.save('./saved_model/pa_repointnet')



print("Done")