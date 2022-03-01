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
from tensorflow.keras.losses import MeanSquaredError
#import tensorflow_model_optimization as tfmo
#from tensorflow.keras.layers import Conv2D
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
#from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Change the working directory
os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN")


# Set the desired lag...
Lag1 = 5


# Read in the data
SSTlonlat = np.genfromtxt("SSTlonlat.dat", delimiter =',')
SoilMoisturelonlat = np.genfromtxt("SoilMoisturelonlat.dat", delimiter =',')
#SSTlonlat_clust = np.genfromtxt("SSTlonlat_clust.dat", delimiter =' ')
sstClust = np.genfromtxt("sstClust.dat", delimiter =',')
SSTanom1 = np.genfromtxt("SSTanom011950_122009clust.dat", delimiter ='\t')
SSTanom2 = np.genfromtxt("SSTanom011950_122009.dat", delimiter =',')
SoilMoist1 = np.genfromtxt("SoilMoisture011950_072010anom.dat", delimiter =',')
LandData1 = np.genfromtxt("Landlonlat.csv", delimiter = ',',skip_header=1)


# Based on these webpages
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/


# Set up a neural network
# define base model
SoilTrain =  np.transpose( SoilMoist1[:,Lag1:(500+Lag1)] )
SSTTrain = np.transpose( SSTanom2[:,0:500] )
SSTFirst = SSTTrain[0,:]
SSTTrainSd1 = np.std(SSTTrain, axis = 0 )

# Build a model
model = Sequential()
model.add( Dense( 1000, input_dim = 3186, activation = 'tanh' ) )
model.add( Dense( 1000, activation = 'tanh' ) )
#model.add( Dense( 500, activation = 'tanh' ) )
model.add( Dense( 328, activation = 'tanh' ) )

# # Compile the model
model.compile( loss = 'mean_squared_error', optimizer='adam')
    
# # fit the keras model on the dataset
model.fit(SSTTrain, SoilTrain, epochs=100)#, batch_size=10)


SSTtest = np.reshape( SSTanom2[:,0], (1, 3186) )
SoilTest = SoilMoist1[:,0]

# Predict with the model
y1 = model.predict( SSTtest )
MSE1 = MeanSquaredError()
y1MSE = MSE1( SoilTest, y1 ).numpy()

# Data for a three-dimensional line
zpoints1 = SoilTrain[0,:]
xpoints1 = SoilMoisturelonlat[:,0]
ypoints1  = SoilMoisturelonlat[:,1]

# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilMoist1[:,501+Lag1,]
SSTFirst = SSTanom2[:,501]
Zsens1 = np.copy( SSTFirst )
Zn1 = np.shape(Zsens1)[0]
Zsens2 = np.copy( SSTFirst )
Zsens3 = np.copy( SSTFirst )
for i in range( 0, Zn1 ):
   Ztest = np.copy( SSTFirst )
   Ztest[i] = Ztest[i] + SSTTrainSd1[i]
   Zp = model.predict( Ztest.reshape([1,Zn1] ) )
   Zsens1[i] = np.std( Zp ) - Zsd1
   Zsens2[i] = np.abs(MSE1( Z1, Zp).numpy() - y1MSE)
   Zsens3[i] = MSE1( Z1, Zp ).numpy()/y1MSE


# Pack it into a data frame
data1 = [ zpoints1, xpoints1, ypoints1 ]
df= pd.DataFrame( data = np.transpose(data1), columns = ['Z','X','Y'] )

Zsens2mx = np.max( Zsens2 )
Zsens2mn = np.min( Zsens2 )
Zsens1s = (1-(Zsens2 - Zsens2mn)/(Zsens2mx - Zsens2mn))
#Zsens2s = np.reshape( Zsens1s, (6,3186))
data2 = np.hstack( (SSTlonlat, np.reshape(Zsens1s, (3186,1))))
df2 = pd.DataFrame( data2 , columns=['X','Y','Z0'] )
Zsens3mx = np.max( Zsens3 )
Zsens3mn = np.min( Zsens3 )
Zsens3s = 1 - (Zsens3 - Zsens3mn)/(Zsens3mx - Zsens3mn)
#Zsens3s1 = np.reshape( Zsens3s, (6,3186) )
data3 =  np.hstack( ( SSTlonlat , np.reshape(Zsens3s, (3186,1) )))
df3 = pd.DataFrame( data3,  columns=['X','Y','Z0'] )




#################################################################################
#  Make some pictures
#

fig = plt.figure()
plt.scatter( LandData1[:,0], LandData1[:,1], s = 1, c = 'black')
plt.scatter( df2.X, df2.Y, s = 5*df2.Z0, c = df2.Z0, cmap = 'Blues') #, alpha = 0.5)
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time '+str(501 + Lag1)+' (Raw) \n Using L=501')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Pred_Lag'+str(Lag1)+'.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( LandData1[:,0], LandData1[:,1], s = 1, c = 'black')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z0, c = df3.Z0, cmap = 'Blues') #, alpha = 0.5)
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time '+str(501 + Lag1)+' (Raw) \n Using L=501')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Pred_Lag'+str(Lag1)+'_Ratio.png', format = 'png')#, quality = 100)
plt.show()





# Write the data out...
df2.to_csv("ANNDerivNoPCALag"+str(Lag1)+".csv", index = False )
df3.to_csv("ANNDerivNoPCALag"+str(Lag1)+"_Ratio.csv", index = False)
