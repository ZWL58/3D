# 3D crack predictionAdd commentMore actions
To achieve accurate 3D crack prediction for thermal compression bondinh (TCB) electrodes, we developed the PA-CVAE model and the PA-RePointNet model.PA-CVAE solves the problem of data scarcity by facilitating the generation of crack data through the positional attention force (PA).PA-RePointNet uses the point cloud and the PA, which allows for accurate prediction of the crack point cloud.
## Requirements
This project is constructed based on the following environment and libraries. To ensure the program runs correctly, please verify that your development environment aligns with the specified configuration listed below:
Python: 3.7.16
TensorFlow: 2.7.0
Cuda: 11.2
Cudnn: 8.9.7
## File structure description
This project includes three main folders: dasets, PA_CVAE and PA_RePointNet.
### datasets
It contains current data and point cloud data. These datasets are important resources for training and testing models.
### PA_CVAE
The main program of the current generation model is located in the main_cvae_pam.py file under the PA_CVAE folder. This model utilizes the conditional variational autoencoder (CVAE) technology to generate analog current data.
### PA_RePointNet
The primary program of the point cloud prediction model is situated in the main.py file within the PA_RePointNet directory. This model is specifically designed for predictive analysis of point cloud data and is applicable to a wide range of 3D data analysis tasks.
By appropriately configuring the environment and utilizing the provide datasets and pre-trained models, it is possible to investigate and expreiment with various scenarios of current generation and point cloud prediction.
