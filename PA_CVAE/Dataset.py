import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import os
from scipy.signal import savgol_filter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.config.list_physical_devices('GPU'))

tf.random.set_seed(123)

dir_path = "./Dataset/"
DatasetName = 'TFRecord-TempCrack-CurTempCurve'

def _bytes_feature(value):
    """Returns a bytes_list from a string/byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from a EagerTensor
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Return a float_list from a float/double."""
    if type(value) != list:
        return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))
    return tf.train.Feature(float_list = tf.train.FloatList(value=value))

def _int64_featrue(value):
    """Return a int64_list from a bool/enum/int/uint."""
    if type(value) != list:
        return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# The user obtains the lower current curve datasets of different temperatures and different defect lengths previously
def Get_Dataset(para_scaling_flag, current_scaling_flag, smooth_flag, sampling_rate=0):
    Para,CurrentArrayAll,TemperatureArray = [],[],[]
    with tf.io.TFRecordWriter(DatasetName) as writer:
        for file in os.listdir(dir_path):                           # Traverse all the txt data files under the folder
            if file == 'readme.txt': continue   # Ignore the readme.txt file
            if (file.split('.')[0]).split('-')[2] != '5': continue

            TempArray,CurrentArray = [],[]

            weld_time = np.linspace(1,1500,3000)    # Welding time
            filename = file.split('.')[0] # file = 'ECA-Temp-Index.txt'
            para = filename.split('-') # file = 'ECA-Temp-Index'

            # Analyze the effective conductive area effective conducting area —— ECA (mm^2)
            if para_scaling_flag == True:
                eca = float(int(para[0])/100.) * 1.6 / 3.2 #  effective conducting length * width = eca (mm^2)
                temp = float(int(para[1].split('C')[0])) / 500.
            else:
                eca = float(int(para[0])/100.) * 1.6 #  effective conducting length * width = eca (mm^2)
                temp = float(int(para[1].split('C')[0]))

            with open(dir_path+file, "r") as f:
                data = f.readline()
                alldata = data.split('FF FF ')
                tTempArray = alldata[1].split(" ")
                tCurrentArray = alldata[2].split(" ")

                for count in range(1, len(tTempArray), 2):
                    if tTempArray[count] == '':
                        break
                    s_tempdata = tTempArray[count] + tTempArray[count + 1]
                    i_tempdata = float(int(s_tempdata, 16))
                    TempArray.append(i_tempdata)

                for count in range(0, len(tCurrentArray), 2):
                    if tCurrentArray[count] == '':
                        break
                    s_tempdata = tCurrentArray[count] + tCurrentArray[count + 1]
                    i_tempdata = float(int(s_tempdata, 16))
                    if current_scaling_flag == True:
                        CurrentArray.append(i_tempdata/2000.)
                    else:
                        CurrentArray.append(i_tempdata)

            if smooth_flag == True:
                CurrentArray = savgol_filter(CurrentArray,window_length=201,polyorder=1)
                CurrentArray = CurrentArray.tolist()
            if sampling_rate != 0:
                CurrentArray = CurrentArray[::sampling_rate]
                # CurrentArray.astype('float32')

            example = (tf.train.Example(features = tf.train.Features(feature={
                'Temp-Crack': _float_feature([eca,temp]),
                # 'Temperature':_int64_featrue(temperature),
                # 'Crack':_float_feature(crack),
                # 'TemperatureCurve':_int64_featrue(TempArray[26:]),
                'CurrentCurve': _float_feature(CurrentArray),
            }))).SerializeToString()

            writer.write(example)

            Para.append([eca, temp]) # eca was reduced by 3.2 times; The temp has been reduced by 500 times
            CurrentArrayAll.append(CurrentArray) # The current value has been reduced by 2,000 times
            TemperatureArray.append(TempArray)

    # minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # Para = minmax_scale.fit_transform(Para)

    # In para, the temperature value is divided by 500 and the crack is divided by 2. The current value in CurrentArrayAll is divided by 2000
    return Para,CurrentArrayAll,TemperatureArray

def Get_Dataset1(dir_path, smooth_flag, sampling_rate=0):

    WeldingParaListAll,CurrentListAll,TempListAll = [],[],[]

    for file in os.listdir(dir_path):

        # File judgment
        if file == 'Initial temperature - Welding temperature - Heating time - welding time - Defect length.txt': continue
        if file.split('.')[-1] != 'txt': continue

        # Analyze the temperature and current data
        TempList, CurrentList = [], []
        with open(dir_path+file, "r") as f:
            data = f.readline()
            alldata = data.split('FF FF ')
            tTempArray = alldata[1].split(" ")
            tCurrentArray = alldata[2].split(" ")

            for count in range(1, len(tTempArray), 2):
                if tTempArray[count] == '':
                    break
                s_tempdata = tTempArray[count] + tTempArray[count + 1]
                i_tempdata = float(int(s_tempdata, 16))
                TempList.append(i_tempdata/500.)

            for count in range(0, len(tCurrentArray), 2):
                if tCurrentArray[count] == '':
                    break
                s_tempdata = tCurrentArray[count] + tCurrentArray[count + 1]
                i_tempdata = float(int(s_tempdata, 16))
                CurrentList.append(i_tempdata/1500.)

        # The curve is smooth.
        if smooth_flag == True:
            CurrentList = savgol_filter(CurrentList,window_length=201,polyorder=1).tolist()
            # CurrentList = CurrentList

        # Uniform sampling of the curve
        if sampling_rate != 0:
            CurrentList = CurrentList[::sampling_rate]
            # CurrentArray.astype('float32')

        # Parse the label -"baseTemp-weldTemp-T1-T2-ECA.csv eg:80-420-500-4000 -025e-2.csv ",T1= heating time, T2= holding time, 025E-2= defect length 0.25mm
        baseTemp = TempList[26] # It was reduced by 500 times
        # weldTemp = int(file.split('-')[1])/500. # The welding temperature is reduced by 500 times
        # time1 = file.split('-')[2] #
        # time2 = file.split('-')[3]
        crackLength = (int((file.split('-')[4]).split('E')[0]))/100. # mm
        eca = (2-crackLength)*1.6/3.2 # effective conducting area —— ECA (mm^2) Reduce by 3.2 times

        WeldingParaListAll.append([baseTemp, eca]) # All temp have been reduced by 500 times; tim1 shrinks by 500 times; time2 has shrunk by 4,000 times. eca was reduced by 3.2 times;
        CurrentListAll.append(CurrentList) # The current value has been reduced by 1,500 times
        TempListAll.append(TempList[26:]) # The temperature value has decreased by 500 times

    # minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # Para = minmax_scale.fit_transform(Para)

    return WeldingParaListAll,CurrentListAll,TempListAll

# WeldingParaListAll,CurrentListAll,TempListAll = Get_Dataset1(dir_path='./420-500-4000 current temperature datasets at initial temperatures of 80-130/', smooth_flag=False, sampling_rate=0)
#
# plt.plot(CurrentListAll[0])
# plt.show()
#
# print("Done")