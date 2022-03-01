#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:42:21 2021

@author: ed
"""

import os
import numpy as np
import pandas as pd
#from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#from mpl_toolkits import mplot3d
#import matplotlib.pyplot as plt

# Change the working directory
os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")


# Read in the data
SSTlonlat = np.genfromtxt("SSTlonlat.dat", delimiter =',')
SoilMoisturelonlat = np.genfromtxt("SoilMoisturelonlat.dat", delimiter =',')
#SSTlonlat_clust = np.genfromtxt("SSTlonlat_clust.dat", delimiter =' ')
sstClust = np.genfromtxt("sstClust.dat", delimiter =',')
SSTanom1 = np.genfromtxt("SSTanom011950_122009clust.dat", delimiter ='\t')
SSTanom2 = np.genfromtxt("SSTanom011950_122009.dat", delimiter =',')
SoilMoist1 = np.genfromtxt("SoilMoisture011950_072010anom.dat", delimiter =',')




# Based on these webpages
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/


# Set up a neural network
# define base model
SoilTrain =  np.transpose( SoilMoist1[:,0:500] )
SSTTrain = np.transpose( SSTanom2[:,0:500] )
SSTFirst = SSTTrain[0,:]
SSTTrainSd1 = np.std(SSTTrain, axis = 0 )

# Build a model
model = Sequential()
model.add( Dense( 1000, input_dim = 3186, activation = 'tanh' ) )
model.add( Dense( 1000, activation = 'tanh' ) )
model.add( Dense( 500, activation = 'tanh' ) )
model.add( Dense( 328, activation = 'tanh' ) )

# # Compile the model
model.compile( loss = 'mean_squared_error', optimizer='adam')
    
# # fit the keras model on the dataset
model.fit(SSTTrain, SoilTrain, epochs=150)#, batch_size=10)

# Predict with the model
y1 = model.predict( SSTTrain )

# Data for a three-dimensional line
zpoints1 = SoilTrain[0,:]
xpoints1 = SoilMoisturelonlat[:,0]
ypoints1  = SoilMoisturelonlat[:,1]

Zsd1 = np.std( SoilTrain[0,:] )
Zsens1 = SSTFirst*0
#Zsens2 = SSTFirst*0
Zn1 = np.shape(Zsens1)[0]
for i in range( 0, Zn1 ):
   Ztest = SSTFirst
   #Ztest2 = SSTFirst
   Ztest[i] = Ztest[i] + SSTTrainSd1[i]
   #Ztest2[i] = Ztest2[i] - SSTTrainSd1[i]
   Zp = model.predict( Ztest.reshape([1,Zn1] ) )
   #Zp2 = model.predict( Ztest2.reshape([1,Zn1] ) )
   Zsens1[i] = np.std( Zp ) - Zsd1
   #Zsens2[i] = np.std( Zp2 ) - Zsd1


# Pack it into a data frame
data1 = [ zpoints1, xpoints1, ypoints1 ]
df= pd.DataFrame( data = np.transpose(data1), columns = ['Z','X','Y'] )

data2 = [ Zsens1, SSTlonlat[:,0], SSTlonlat[:,1] ]
df2 = pd.DataFrame( data = np.transpose(data2), columns=['Z','X','Y'] )
#data3 = [ Zsens2, SSTlonlat[:,0], SSTlonlat[:,1] ]
#df3 = pd.DataFrame( data = np.transpose(data3), columns=['Z','X','Y'] )


# fig = plt.figure()
# plt.scatter( df2.X, df2.Y, s-=50, c = df.Z, cmap = 'gray')

# Write the data out...
df2.to_csv("ANNDeriv.csv", index = False )

