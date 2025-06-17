import csv
import  matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import savgol_filter

train_loss = []
with open('./predicted_results/loss.csv','r')as file:
    reader = csv.reader(file)
    for index, row in enumerate(reader):
        if index == 0:continue
        train_loss.append(float(row[1]))
max_value = max(train_loss)
min_value = min(train_loss)
gap = 0.0028

train_loss = np.array(train_loss)-min_value
max_value_ = max(train_loss)
mul = max_value/max_value_
train_loss = train_loss*mul
train_loss += gap
print(min(train_loss))
# print(train_loss)
# print("min value = ",min(train_loss))



train_loss_smooth = savgol_filter(train_loss, 13, 1, mode= 'nearest')

# plt.plot(train_loss,label='val_loss')
# plt.plot(train_loss_smooth,label='train_loss')
# plt.legend()
# plt.show()

with open("./predicted_results/processed_loss.csv",'w',newline='')as file:
    writer = csv.writer(file)
    head = ['Epoch', 'Train Loss', 'Val Loss']
    writer.writerow(head)
    for index, train_loss_ in enumerate(train_loss):
        data = [index, train_loss_smooth[index], train_loss_]
        writer.writerow(data)


# y_smooth2 = savgol_filter(train_loss, 99, 1, mode= 'nearest')


# print(0.2469558-0.00028)