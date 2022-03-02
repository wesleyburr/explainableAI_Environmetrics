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
#from sklearn.preprocessing import scale
#from sklearn.pipeline import Pipeline
#from mpl_toolkits import mplot3d
#import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats as sc

# Change the working directory
# os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN")
# os.chdir("/home/ed/Documents/TiesWG")
# os.chdir("/home/ed/Documents/GitHub/explainableAI_Environmetrics/ANN")
os.chdir("/Users/ed/Documents/GitHub/Spatial/explainableAI_Environmetrics/ANN")

# Set the desired lag...
Lag1 = 0


# Read in the data
SST1_data1 =  pd.read_csv("SST_data.csv");
SSTlonlat = SST1_data1[ ( ['Lon','Lat'] ) ];
SST1_data2 = np.asarray(SST1_data1);
SSTanom2 = SST1_data2[ :, 3:891 ];
#SSTlonlat = np.genfromtxt("SSTlonlat.dat", delimiter =',')
#SoilMoisturelonlat = np.genfromtxt("SoilMoisturelonlat.dat", delimiter =',')
#SSTlonlat_clust = np.genfromtxt("SSTlonlat_clust.dat", delimiter =' ')
#sstClust = np.genfromtxt("sstClust.dat", delimiter =',')
#SSTanom1 = np.genfromtxt("SSTanom011950_122009clust.dat", delimiter ='\t')
#SSTanom2 = np.genfromtxt("SSTanom011950_122009.dat", delimiter =',')
SoilMoist1in = pd.read_csv("cornbelt.csv", sep =',', header = 0 )
LandData1 = SoilMoist1in[ (['Lon','Lat']) ];
SoilMoist1a = np.asarray(SoilMoist1in);
SoilMoist1b = SoilMoist1a[:,3:890]
SoilMoist1bm = np.mean( SoilMoist1b, axis = 0 )
SoilMoist1s = np.std( SoilMoist1b, axis = 0 )
SoilMoist1c = ( SoilMoist1b - SoilMoist1bm )/SoilMoist1s
SoilMoist1 = SoilMoist1c;


# Based on these webpages
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/




pca1 = PCA( n_components = 60 )
pca1.fit( SSTanom2[:,0:784].T )
plt.grid()
plt.plot( np.cumsum( pca1.explained_variance_ratio_ * 100 ) ) 

# Set up a neural network
# define base model
Soil1 = np.zeros(1224)
SST1 = np.zeros( [1, 60*6] )
SST1Raw = np.zeros([ 1, 3186*6 ])

for i in range(780):
    SoilTraintmp = SoilMoist1[:,(i+9)]
    Soil1 = np.vstack( (Soil1, SoilTraintmp) )
    SSTtmp1 = np.concatenate(( pca1.transform( SSTanom2[:,i].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+1)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+2)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+3)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+4)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+5)].reshape([1,3186])).T))
    SST1RawTmp1 = np.concatenate( ( SSTanom2[:,i].reshape([1,3186]).T,
                                    SSTanom2[:,(i+1)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+2)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+3)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+4)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+5)].reshape([1,3186]).T
                                                          )) 
    SST1 = np.vstack( (SST1, SSTtmp1.T))
    SST1Raw = np.vstack( (SST1Raw, SST1RawTmp1.T))
    
Soil2 = np.delete( Soil1, 0, axis = 0 )
SST2v = np.delete( SST1, 0, axis = 0 )
SST1Raw = np.delete( SST1Raw, 0, axis = 0 )

# Run the PCA version 
    
SoilTrain =  np.copy( Soil2 )
SSTTrain = np.copy( SST2v )
SSTFirst = SSTTrain[0,:]
SSTTrainSd1 = np.std(SST1Raw, axis = 0 )
SoilTesta = SoilMoist1a[:,797]

# Build a model
model = Sequential()
model.add( Dense( 420, input_dim = 60*6, activation = 'tanh' ) )
model.add( Dense( 420, activation = 'tanh' ) )
#model.add( Dense( 420, activation = 'tanh' ) )
model.add( Dense( 1224))#, activation = 'tanh' ) )

# # Compile the model
model.compile( loss = 'mean_squared_error', optimizer='adam')
    
# # fit the keras model on the dataset
model.fit(SSTTrain, SoilTrain, epochs=500)#, batch_size=10)


# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilMoist1[:,797]
SSTFirst = np.concatenate( ( pca1.transform(SSTanom2[:,789].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,790].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,791].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,792].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,793].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,794].reshape([1,3186])).T ) )
# Predict with the model
y1 = model.predict( np.reshape( SSTFirst, (1, 60*6) ) )
y1a = y1*SoilMoist1s[797] + SoilMoist1bm[797]
MSE1 = MeanSquaredError()
y1MSE = MSE1( Z1, y1 ).numpy()
y1MSEa = MSE1( SoilTesta, y1a ).numpy()
plt.figure()
plt.plot( y1.T, Z1, 'o' )
predR2 = sc.pearsonr(y1.T[:,0], Z1)[0]**2



# Data for a three-dimensional line
zpoints1 = SoilTrain[0,:]
xpoints1 = LandData1['Lat']
ypoints1  = LandData1['Lon']


# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilMoist1[:,797]
SSTFirst = np.concatenate( (SSTanom2[:,789],
    SSTanom2[:,790],
    SSTanom2[:,791],
    SSTanom2[:,792],
    SSTanom2[:,793],
    SSTanom2[:,794]) )
Zsens1 = np.copy( SSTFirst )
Zn1 = np.shape(Zsens1)[0]
Zsens2 = np.copy( SSTFirst )
Zsens3 = np.copy( SSTFirst )
for i in range( 0, Zn1):
   Ztest = np.copy( SSTFirst )
   Ztest[i] = Ztest[i] + SSTTrainSd1[i]
   Ztest2 = Ztest.reshape([6,3186])
   Ztest3 = np.concatenate(( pca1.transform(Ztest2[0,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[1,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[2,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[3,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[4,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[5,:].reshape([1,3186])).T ))
   Zp = model.predict( Ztest3.T ) 
   Zsens1[i] = np.std( Zp ) - Zsd1
   Zsens2[i] = np.abs(MSE1( Z1, Zp).numpy() - y1MSE)
   Zsens3[i] = MSE1( Z1, Zp ).numpy()/y1MSE



# Pack it into a data frame
data1 = [ zpoints1, xpoints1, ypoints1 ]
df= pd.DataFrame( data = np.transpose(data1), columns = ['Z','X','Y'] )

#Zsens2mx = np.max( Zsens2 )
#Zsens2mn = np.min( Zsens2 )
#Zsens1s = (1-(Zsens2 - Zsens2mn)/(Zsens2mx - Zsens2mn))
#Zsens2s = np.reshape( Zsens1s, (7,3186))
#data2 = np.hstack( (SSTlonlat, Zsens2s.T ))
#df2 = pd.DataFrame( data2 , columns=['X','Y','Z1','Z2','Z3','Z4','Z5','Z6','Z7'] )
#Zsens3mx = np.max( Zsens3 )           
#Zsens3mn = np.min( Zsens3 )
#Zsens3s = 1 - (Zsens3 - Zsens3mn)/(Zsens3mx - Zsens3mn)
Zsens3s1 = np.reshape( Zsens3, (6,3186))
data3 =  np.hstack( (SSTlonlat, Zsens3s1.T ))
df3 = pd.DataFrame( data3,  columns=['X','Y','Z1','Z2','Z3','Z4','Z5','Z6'] )
                    
# Write the data out...
df3.to_csv("Plots/ANNDerivWideRatioPCA.csv", index = False )

##############################################################################
# Make the plots - Based on Ratio
#
##############################################################################

fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z6, c = df3.Z6, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('February to May (PCA$_{60}$ Wide) February')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_Feb_to_May_PCA_ratio.png', format = 'png')#, quality = 100)
plt.show()

fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z5, c = df3.Z5, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('February to May (PCA$_{60}$ Wide) January')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_Jan_to_May_PCA_ratio.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z4, c = df3.Z4, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('February to May (PCA$_{60}$ Wide) December')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_Dec_to_May_PCA_ratio.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z3, c = df3.Z3, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('February to May (PCA$_{60}$ Wide) November')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_Nov_to_May_PCA_ratio.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z2, c = df3.Z2, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Feburary to May (PCA$_{60}$ Wide) October')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_Oct_to_May_PCA_ratio.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z1, c = df3.Z1, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('February to May (PCA$_{60}$ Wide) September')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_Sep_to_May_PCA_ratio.png', format = 'png')#, quality = 100)
plt.show()






#fig = plt.figure()
#plt.scatter( df3.X, df3.Y, s = df3.Zs)#, alpha = 0.5)
#plt.savefig('PredErr_Lag'+str(Lag1)+'out.svg', format = 'svg')#, quality = 100)


