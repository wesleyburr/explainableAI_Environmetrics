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
from sklearn.preprocessing import scale
#from sklearn.pipeline import Pipeline
#from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats as sc

# Change the working directory
# os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN")
os.chdir("/home/ed/Documents/TiesWG/Model_ANN")

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
SoilMoist1in = pd.read_csv("Soil.csv", sep =',', header = 0 )
LandData1 = SoilMoist1in[ (['Lon','Lat']) ];
SoilMoist1a = np.asarray(SoilMoist1in);
SoilMoist1b = SoilMoist1a[:,2:890]
SoilMoist1c = scale( SoilMoist1b, axis = 0 );
SoilMoist1 = SoilMoist1c;
Month1 = pd.read_csv("MonInd1.csv", sep =',', header = 0 )
Month2 = np.asarray( Month1[(['Jan','Feb','Mar','April','May','June','July','August','September ','October','November'])] )

# Based on these webpages
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/




pca1 = PCA( n_components = 60 )
pca1.fit( SSTanom2[:,0:784].T )
plt.grid()
plt.plot( np.cumsum( pca1.explained_variance_ratio_ * 100 ) ) 

# Set up a neural network
# define base model
Soil1 = np.zeros(4587)
SST1 = np.zeros( [1, 60*7] )
SST1Raw = np.zeros([ 1, 3186*7 ])

for i in range(784):
    SoilTraintmp = SoilMoist1[:,(i+7)]
    Soil1 = np.vstack( (Soil1, SoilTraintmp) )
    SSTtmp1 = np.concatenate(( pca1.transform( SSTanom2[:,i].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+1)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+2)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+3)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+4)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i+5)].reshape([1,3186])).T,
        pca1.transform(SSTanom2[:,(i + 6)].reshape([1,3186])).T))
    SST1RawTmp1 = np.concatenate( ( SSTanom2[:,i].reshape([1,3186]).T,
                                    SSTanom2[:,(i+1)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+2)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+3)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+4)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+5)].reshape([1,3186]).T,
                                    SSTanom2[:,(i+6)].reshape([1,3186]).T
                                                          )) 
    SST1 = np.vstack( (SST1, SSTtmp1.T))
    SST1Raw = np.vstack( (SST1Raw, SST1RawTmp1.T))
    
    
Soil2 = np.delete( Soil1, 0, axis = 0 )
SST2v = np.delete( SST1, 0, axis = 0 )
SST1Raw = np.delete( SST1Raw, 0, axis = 0 )
SST2v = np.hstack( (SST2v, Month2[0:784,:]) )

# Run the PCA version 
    
SoilTrain =  np.copy( Soil2 )
SSTTrain = np.copy( SST2v )
SSTFirst = SSTTrain[0,:]
SSTTrainSd1 = np.std(SST1Raw, axis = 0 )

# Build a model
model = Sequential()
model.add( Dense( 420, input_dim = 60*7+11, activation = 'tanh' ) )
model.add( Dense( 400, activation = 'tanh' ) )
model.add( Dense( 350, activation = 'tanh' ) )
model.add( Dense( 4587))#, activation = 'tanh' ) )

# # Compile the model
model.compile( loss = 'mean_squared_error', optimizer='adam')
    
# # fit the keras model on the dataset
model.fit(SSTTrain, SoilTrain, epochs=1000)#, batch_size=10)


# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilMoist1[:,800]
SSTFirst = np.concatenate( ( pca1.transform(SSTanom2[:,793].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,794].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,795].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,796].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,797].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,798].reshape([1,3186])).T,
                             pca1.transform(SSTanom2[:,799].reshape([1,3186])).T,
                             Month2[799,:].reshape(([1,11])).T ) )
# Predict with the model
y1 = model.predict( np.reshape( SSTFirst, (1, 60*7+11) ) )
MSE1 = MeanSquaredError()
y1MSE = MSE1( Z1, y1 ).numpy()
plt.figure()
plt.plot( y1.T, Z1, 'o' )
np.correlate(y1.T[:,0], Z1)**2
predR2 = sc.pearsonr(y1.T[:,0], Z1)[0]**2
# 0.6483997919926561


# Data for a three-dimensional line
zpoints1 = SoilTrain[0,:]
xpoints1 = LandData1['Lat']
ypoints1  = LandData1['Lon']


# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilMoist1[:,800]
SSTFirst = np.concatenate( (SSTanom2[:,793],
    SSTanom2[:,794],
    SSTanom2[:,795],
    SSTanom2[:,796],
    SSTanom2[:,797],
    SSTanom2[:,798],
    SSTanom2[:,799]) )
Zsens1 = np.copy( SSTFirst )
Zn1 = np.shape(Zsens1)[0]
Zsens2 = np.copy( SSTFirst )
Zsens3 = np.copy( SSTFirst )
for i in range( 0, Zn1):
   Ztest = np.copy( SSTFirst )
   Ztest[i] = Ztest[i] + SSTTrainSd1[i]
   Ztest2 = Ztest.reshape([7,3186])
   Ztest3 = np.concatenate(( pca1.transform(Ztest2[0,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[1,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[2,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[3,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[4,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[5,:].reshape([1,3186])).T,
                            pca1.transform(Ztest2[6,:].reshape([1,3186])).T,
                            Month2[799,:].reshape(([1,11])).T ) )
   Zp = model.predict( Ztest3.T ) 
   Zsens1[i] = np.std( Zp ) - Zsd1
   Zsens2[i] = np.abs(MSE1( Z1, Zp).numpy() - y1MSE)
   Zsens3[i] = MSE1( Z1, Zp ).numpy()/y1MSE



# Pack it into a data frame
data1 = [ zpoints1, xpoints1, ypoints1 ]
df= pd.DataFrame( data = np.transpose(data1), columns = ['Z','X','Y'] )

Zsens2mx = np.max( Zsens2 )
Zsens2mn = np.min( Zsens2 )
Zsens1s = (1-(Zsens2 - Zsens2mn)/(Zsens2mx - Zsens2mn))
Zsens2s = np.reshape( Zsens1s, (7,3186))
data2 = np.hstack( (SSTlonlat, Zsens2s.T ))
df2 = pd.DataFrame( data2 , columns=['X','Y','Z1','Z2','Z3','Z4','Z5','Z6','Z7'] )
Zsens3mx = np.max( Zsens3 )           
Zsens3mn = np.min( Zsens3 )
Zsens3s = 1 - (Zsens3 - Zsens3mn)/(Zsens3mx - Zsens3mn)
Zsens3s1 = np.reshape( Zsens3s, (7,3186))
data3 =  np.hstack( (SSTlonlat, Zsens3s1.T ))
df3 = pd.DataFrame( data3,  columns=['X','Y','Z1','Z2','Z3','Z4','Z5','Z6','Z7'] )
                    
##############################################################################
# Make the plots
#
##############################################################################

# fig = plt.figure()
# plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
# plt.scatter( df2.X, df2.Y, s = 5*df2.Z1, c = df2.Z1, cmap = 'Blues') #, alpha = 0.5)
# plt.colorbar()
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.title('Time 550 (PCA) 549 \n Using Time 543 to 549')
# #os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
# plt.savefig('Plots/Pred550from549PCA.png', format = 'png')#, quality = 100)
# plt.show()

# fig = plt.figure()
# plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
# plt.scatter( df2.X, df2.Y, s = 5*df2.Z2, c = df2.Z2, cmap = 'Blues') #, alpha = 0.5)
# plt.colorbar()
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.title('Time 550 (PCA) 548 \n Using Time 543 to 549')
# plt.savefig('Plots/Pred550from548PCA.png', format = 'png')#, quality = 100)
# plt.show()

# fig = plt.figure()
# plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
# plt.scatter( df2.X, df2.Y, s = 5*df2.Z3, c = df2.Z3, cmap = 'Blues') #, alpha = 0.5)
# plt.colorbar()
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.title('Time 550 (PCA) 547 \n Using Time 543 to 549')
# plt.savefig('Plots/Pred550from547PCA.png', format = 'png')#, quality = 100)
# plt.show()


# fig = plt.figure()
# plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
# plt.scatter( df2.X, df2.Y, s = 5*df2.Z4, c = df2.Z4, cmap = 'Blues') #, alpha = 0.5)
# plt.colorbar()
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.title('Time 550 (PCA) 546 \n Using Time 543 to 549')
# plt.savefig('Plots/Pred550from546PCA.png', format = 'png')#, quality = 100)
# plt.show()


# fig = plt.figure()
# plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
# plt.scatter( df2.X, df2.Y, s = 5*df2.Z5, c = df2.Z5, cmap = 'Blues') #, alpha = 0.5)
# plt.colorbar()
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.title('Time 550 (PCA) 545 \n Using Time 543 to 549')
# plt.savefig('Plots/Pred550from545PCA.png', format = 'png')#, quality = 100)
# plt.show()


# fig = plt.figure()
# plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
# plt.scatter( df2.X, df2.Y, s = 5*df2.Z6, c = df2.Z6, cmap = 'Blues') #, alpha = 0.5)fig = plt.figure()
# plt.colorbar()
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.title('Time 550 (PCA) 544 \n Using Time 543 to 549')
# plt.savefig('Plots/Pred550from544PCA.png', format = 'png')#, quality = 100)
# plt.show()

# fig = plt.figure()
# plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
# plt.scatter( df2.X, df2.Y, s = 5*df2.Z7, c = df2.Z7, cmap = 'Blues') #, alpha = 0.5)fig = plt.figure()
# plt.colorbar()
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.title('Time 550 (PCA) 543 \n Using Time 543 to 549')
# plt.savefig('Plots/Pred550from543PCA.png', format = 'png')#, quality = 100)
# plt.show()



##############################################################################
# Make the plots - Based on Ratio
#
##############################################################################

fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z1, c = df3.Z1, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time 800 (PCA) 799 \n Using Time 793 to 799')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred800from799PCA_ratioM.png', format = 'png')#, quality = 100)
plt.show()

fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z2, c = df3.Z2, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time 800 (PCA) 798 \n Using Time 793 to 799')
plt.savefig('Plots/Pred800from798PCA_ratioM.png', format = 'png')#, quality = 100)
plt.show()

fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z3, c = df3.Z3, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time 800 (PCA) 797 \n Using Time 793 to 799')
plt.savefig('Plots/Pred800from797PCA_ratioM.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z4, c = df3.Z4, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time 800 (PCA) 796 \n Using Time 793 to 799')
plt.savefig('Plots/Pred800from796PCA_ratioM.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z5, c = df3.Z5, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time 800 (PCA) 795 \n Using Time 793 to 799')
plt.savefig('Plots/Pred800from795PCA_ratioM.png', format = 'png')#, quality = 100)
plt.show()


fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z6, c = df3.Z6, cmap = 'bwr') #, alpha = 0.5)fig = plt.figure()
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time 800 (PCA) 794 \n Using Time 793 to 799')
plt.savefig('Plots/Pred800from794PCA_ratioM.png', format = 'png')#, quality = 100)
plt.show()

fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z7, c = df3.Z7, cmap = 'bwr') #, alpha = 0.5)fig = plt.figure()
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Time 800 (PCA) 793 \n Using Time 793 to 799')
plt.savefig('Plots/Pred800from793PCA_ratioM.png', format = 'png')#, quality = 100)
plt.show()



#fig = plt.figure()
#plt.scatter( df3.X, df3.Y, s = df3.Zs)#, alpha = 0.5)
#plt.savefig('PredErr_Lag'+str(Lag1)+'out.svg', format = 'svg')#, quality = 100)

# Write the data out...
df2.to_csv("Plots/ANNDerivWidePCAM.csv", index = False )
df3.to_csv("Plots/ANNDerivWideRatioPCAM.csv", index = False )
