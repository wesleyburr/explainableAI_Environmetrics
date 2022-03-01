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
#import tensorflow_model_optimization as tfmo
#from tensorflow.keras.layers import Conv2D
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
#from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Change the working directory
os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")


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


# Based on these webpages
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/


# Set up a neural network
# define base model
Soil1 = np.zeros(328)
SST1 = np.zeros( 19116)
for i in range(500):
    SoilTraintmp = SoilMoist1[:,(i+5)]
    Soil1 = np.vstack( (Soil1, SoilTraintmp) )
    SSTtmp1 = np.concatenate(( SSTanom2[:,i],
                              SSTanom2[:,(i+1)],
                              SSTanom2[:,(i+2)],
                              SSTanom2[:,(i+3)],
                              SSTanom2[:,(i+4)],
                              SSTanom2[:,(i + 5)]))
    SST1 = np.vstack( (SST1, SSTtmp1))
    
Soil2 = np.delete( Soil1,0,axis = 0 )
SST2v = np.delete( SST1, 0, axis = 0 )

# Run the PCA version 
    
SoilTrain =  np.copy( Soil2 )
SSTTrain = np.copy( SST2v )
SSTFirst = SSTTrain[0,:]
SSTTrainSd1 = np.std(SSTTrain, axis = 0 )

# Build a model
model = Sequential()
model.add( Dense( 1000, input_dim = 19116, activation = 'tanh' ) )
model.add( Dense( 1000, activation = 'tanh' ) )
#model.add( Dense( 500, activation = 'tanh' ) )
model.add( Dense( 328, activation = 'tanh' ) )

# # Compile the model
model.compile( loss = 'mean_squared_error', optimizer='adam')
    
# # fit the keras model on the dataset
model.fit(SSTTrain, SoilTrain, epochs=1000)#, batch_size=10)

# Predict with the model
y1 = model.predict( SSTTrain )

# Data for a three-dimensional line
zpoints1 = SoilTrain[0,:]
xpoints1 = SoilMoisturelonlat[:,0]
ypoints1  = SoilMoisturelonlat[:,1]

# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilTrain[0,:]
Zsens1 = np.copy( SSTFirst )
Zn1 = np.shape(Zsens1)[0]
Zsens2 = np.copy( SSTFirst )
for i in range( 0, Zn1 ):
   Ztest = np.copy( SSTFirst )
   Ztest[i] = Ztest[i] + SSTTrainSd1[i]
   Zp = model.predict( Ztest.reshape([1,Zn1] ) )
   Zsens1[i] = np.std( Zp ) - Zsd1
   Zsens2[i] = np.sum( np.square( Zp.T - Z1 ) )/np.shape(Zp)[1]


# Pack it into a data frame
data1 = [ zpoints1, xpoints1, ypoints1 ]
df= pd.DataFrame( data = np.transpose(data1), columns = ['Z','X','Y'] )

Zsens2mx = np.max( Zsens2 )
Zsens2mn = np.min( Zsens2 )
Zsens1s = (1-(Zsens2 - Zsens2mn)/(Zsens2mx - Zsens2mn))
Zsens2s = np.reshape( Zsens1s, (6,3186))
data2 = np.hstack( (SSTlonlat, Zsens2s.T ))
df2 = pd.DataFrame( data2 , columns=['X','Y','Z0','Z1','Z2','Z3','Z4','Z5'] )
#Zsens2mx = np.max( Zsens2s )
#Zsens2mn = np.min( Zsens2s )
#Zsens2s = 1 - (Zsens2 - Zsens2mn)/(Zsens2mx - Zsens2mn)
#data3 = [ Zsens2, Zsens2s, SSTlonlat[:,0], SSTlonlat[:,1] ]
#df3 = pd.DataFrame( data = np.transpose(data3), columns=['Z', 'Zs', 'X', 'Y'] )

#Zsens2sT = Zsens2s.T

Xlontmp1 = list( range(124,292,2) )
Xlattmp1 = list( range(-30, 62, 2) )

Xlatlon1 = np.ones( (np.shape(Xlontmp1)[0]*np.shape(Xlattmp1)[0],3) )  
count1 = 0


                      




fig = plt.figure()
plt.scatter( Xlatlon1[:,0], Xlatlon1[:,1], c = 'black', s = 1)
plt.scatter( df2.X, df2.Y, c = 'white', s = 1 )
plt.scatter( df2.X, df2.Y, s = 3, c = df2.Z0, cmap = 'bwr') #, alpha = 0.5)
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Lag 0 \n Using L=0 to 5')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN/Pred0_L0to5out.png', format = 'png')#, quality = 100)
plt.show()

fig = plt.figure()
plt.scatter( df2.X, df2.Y, s = df2.Z1, c = df2.Z1, cmap = 'bwr') #, alpha = 0.5)
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Lag 1 \n Using L=0 to 5')
plt.savefig('/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN/Pred1_L0to5out.png', format = 'png')#, quality = 100)
plt.show()

fig = plt.figure()
plt.scatter( df2.X, df2.Y, s = df2.Z2, c = df2.Z2, cmap = 'bwr') #, alpha = 0.5)
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Lag 2 \n Using L=0 to 5')
plt.savefig('/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN/Pred2_L0to5out.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df2.X, df2.Y, s = df2.Z3, c = df2.Z3, cmap = 'bwr') #, alpha = 0.5)
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Lag 3 \n Using L=0 to 5')
plt.savefig('/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN/Pred3_L0to5out.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df2.X, df2.Y, s = df2.Z4, c = df2.Z4, cmap = 'bwr') #, alpha = 0.5)
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Lag 4 \n Using L=0 to 5')
plt.savefig('/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN/Pred4_L0to5out.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df2.X, df2.Y, s = df2.Z5, c = df2.Z5, cmap = 'bwr') #, alpha = 0.5)fig = plt.figure()
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Lag 5 \n Using L=0 to 5')
plt.savefig('/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN/Pred5_L0to5out.png', format = 'png')#, quality = 100)
plt.show()








#fig = plt.figure()
#plt.scatter( df3.X, df3.Y, s = df3.Zs)#, alpha = 0.5)
#plt.savefig('PredErr_Lag'+str(Lag1)+'out.svg', format = 'svg')#, quality = 100)

# Write the data out...
df2.to_csv("ANNDerivWide.csv", index = False )

