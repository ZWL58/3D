import dataset as DS
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from dataset import *

tf.random.set_seed(123)


def get_dataset():
    ecas, philsms = Get_eca_philsm()
    eca = [ecas[0], ecas[4], ecas[12], ecas[15], ecas[17], ecas[30]]
    # philsm = [philsms[0], philsms[4], philsms[12], philsms[15], philsms[17], philsms[30]]
    philsm = [0, 1, 2, 3, 4, 5]

    pacvae = keras.models.load_model('../CVAE/saved_model/pacvae')
    basetemp = 80
    wps = []
    for index, eca_ in enumerate(eca):
        wp = [basetemp / 500., eca_ / 3.2]
        wps.append(wp)

    z = keras.backend.random_normal(shape=(np.array(wps).shape[0], 20))
    input = tf.concat([z, wps], axis=1)
    current = pacvae.decoder.predict(input)  # input as [z, [baseTemp,eca]]
    return current,np.array(philsm)/5.
# current,philsm = get_dataset()
# print(philsm)

# currents,philsms = get_current_philsm()
# print("OK")

# return ecas, philsms, currents, philsms
#
# print(len(currents[0]))
# print(len(philsms[0]))
#
# plt.plot(currents[0])
# plt.show()