import keras
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import time
from gpu_info import *

from dataset import *

physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

tf.random.set_seed(123)

# Parameter setting
batch_size = 50

gpu_info()

# Test dataset
currents,philsms = get_current_philsm() # dataset.py
print("dataset size: ", len(currents))
data_size = len(currents)
train_dataset_size = int(data_size*0.8)
val_dataset_size = data_size-int(data_size*0.8)
dataset = tf.data.Dataset.from_tensor_slices((currents, philsms)).shuffle(len(currents)).batch(batch_size)

gpu_info()
# time.sleep()

# Loading model test
pa_repointnet = keras.models.load_model('./saved_model/pa_repointnet')

gpu_info()

time_list = []
for i in range(5):
    for data_batch in dataset:
        current = data_batch[0]
        philsm = data_batch[1]
        start_time = time.time()
        predicted_philsm = pa_repointnet.predict(current)
        end_time = time.time()-start_time
        time_list.append(end_time)

gpu_info()

# print(time_list)
plt.plot(time_list)
plt.savefig(f'./predicted_results/inference_time_test_batch{batch_size}.png', bbox_inches='tight')
plt.cla()
plt.close()

with open(f"./predicted_results/inference_time_test_batch{batch_size}.csv",'w',newline='')as file:
    writer = csv.writer(file)
    for data in time_list:
        writer.writerow([data])