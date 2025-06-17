import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import csv
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd

# print(tf.config.list_physical_devices('GPU'))

tf.random.set_seed(123)

def Get_eca_philsm():

    # Obtain eca
    ecas = []
    with open('./eca_dataset/Model-N-Crack-60-420-500-4000-InitCrack-01E-1mm_eca.csv','r')as file:
        reader = csv.reader(file)
        for row in reader:
            ecas.append(float(row[1]))

    # Obtain the longest number of nodes and the node names
    nodelabels = []
    with open('./philsm_dataset/Model-N-Crack-60-420-500-4000-InitCrack-01E-1mm_NodeLabel-Philsm_frame43.csv','r')as file:
        reader = csv.reader(file)
        for index,row in enumerate(reader):
            if index== 0:continue
            nodelabels.append(int(row[0]))
    nodelabels = sorted(nodelabels)

    # Obtain the node philsm data
    philsm = []
    files = os.listdir('./philsm_dataset/')
    for file_index,file in enumerate(files):
        # if index> 1:break
        # Prefabricate a dictionary of all zeros, with the key being nodelabel
        tempdata = {}
        for nodelabel in nodelabels: # Since nodelabels are sorted, the dictionary tempdata does not need to be sorted
            tempdata[nodelabel] = 0
        # Read the csv file data and press the key to update tempdata
        with open(f'./philsm_dataset/{file}', 'r') as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                if index == 0: continue
                nodelabel = int(row[0])
                temp_philsm = float(row[1])
                tempdata[nodelabel] = temp_philsm
        philsm.append(list(tempdata.values()*1000))
    return ecas, philsm

def get_current_philsm():
    ecas, philsm = Get_eca_philsm()
    # pa-cvae-decoder: [z,weld_para]=[z,[baseTemp, eca]] -> current
    # pa-repointnet : Current curve -> philsm
    pacvae = keras.models.load_model('../CVAE/saved_model/pacvae')

    currents,philsms = [],[]
    basetemps = list(range(80, 135, 5))
    for basetemp in basetemps:
        for index,eca in enumerate(ecas):
            wp = [basetemp/500., eca/3.2]
            wp = np.expand_dims(wp,axis=0)
            print(np.array(wp).shape[1])
            z = keras.backend.random_normal(shape=(1, 20))
            input = tf.concat([z, wp], axis=1)
            current = pacvae.decoder.predict(input)  # Input as [z, [baseTemp,eca]]
            for i in range(1):
                currents.append(current[0])
                philsms.append(index/43.)
    del pacvae
    print("ok")
    return currents,philsms

# currents,philsms = Get_Dataset()
#
# print(len(currents))
# print(len(philsms))


