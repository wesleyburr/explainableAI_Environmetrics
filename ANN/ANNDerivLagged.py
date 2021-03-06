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
os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN")
# os.chdir("/home/ed/Documents/TiesWG")

# Set the desired lag...
Lag1 = 4


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
SoilMoist1bm = np.mean( SoilMoist1b, axis = 0 )
SoilMoist1s = np.std( SoilMoist1b, axis = 0 )
SoilMoist1c = ( SoilMoist1b - SoilMoist1bm )/SoilMoist1s
SoilMoist1 = SoilMoist1c;

# "X8.1.2014"


# Based on these webpages
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/


# Set up a neural network
# define base model
SoilTrain =  np.transpose( SoilMoist1[:,Lag1:(784+Lag1)] )
SSTTrain = np.transpose( SSTanom2[:,0:784] )
SSTFirst = SSTTrain[0,:]
SSTTrainSd1 = np.std(SSTTrain, axis = 0 )
SoilTest = SoilMoist1[:,800]
SoilTesta = SoilMoist1a[:,800]


#pca_top1 = PCA( n_components = 60 )
#pca_top1.fit( SSTTrain) 
#STTrain_pca = pca_top1.transform( SSTTrain )

#plt.grid()
#plt.plot( np.cumsum( pca_top1.explained_variance_ratio_ * 100 ) ) 



# Build a model
model = Sequential()
model.add( Dense( 3186, input_dim = 3186, activation = 'tanh' ) )
model.add( Dense( 2000, activation = 'tanh' ) )
model.add( Dense( 2000, activation = 'tanh' ) )
model.add( Dense( 4587 ) )#, activation = 'tanh' ) )

# # Compile the model
model.compile( loss = 'mean_squared_error', optimizer='adam')
    
# # fit the keras model on the dataset
model.fit(SSTTrain, SoilTrain, epochs=100)#, batch_size=10)


SSTtest = np.reshape( SSTanom2[:,(800-Lag1)], (1, 3186) ) 
SoilTest = SoilMoist1[:,800]

# Predict with the model
y1a = model.predict( SSTtest )*SoilMoist1s[800] + SoilMoist1bm[800] 
y1 = model.predict( SSTtest ) 
MSE1 = MeanSquaredError()
y1MSE = MSE1( SoilTest, y1 ).numpy()
y1MSEa = MSE1( SoilTesta, y1a ).numpy()
plt.plot( SoilTest, y1.T ,'o')
predR2 = sc.pearsonr(y1.T[:,0], SoilTest)[0]**2

# Data for a three-dimensional line
zpoints1 = SoilTrain[0,:]
xpoints1 = LandData1['Lat']
ypoints1  = LandData1['Lon']

# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilMoist1[:,800]
SSTFirst = SSTanom2[:,(800-Lag1)]
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

#fig = plt.figure()
#plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
#plt.scatter( df2.X, df2.Y, s = 5*df2.Z0, c = df2.Z0, cmap = 'Blues') #, alpha = 0.5)
#plt.colorbar()
#plt.xlabel('lon')
#plt.ylabel('lat')
#plt.title('Predicted Time '+str(550)+' (PCA) \n Data Time='+str(550-Lag1)+' MSEpred='+str(np.round( y1MSE,3)) )
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
#plt.savefig('Plots/Pred_LagPCA'+str(Lag1)+'.png', format = 'png')#, quality = 100)
#plt.show()


fig = plt.figure()
plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z0, c = df3.Z0, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('August 2014 (RAW) \n Predictive Data: April 2014 \n $R^2_{pred}$='+str(np.round( predR2,3))+', MSE$_{pred}$=' +str(np.round( y1MSEa,3) ))
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_LagRAW'+str(Lag1)+'_Ratio.png', format = 'png')#, quality = 100)
plt.show()



# Write the data out...
df2.to_csv("Plots/ANNDerivRAWLag"+str(Lag1)+".csv", index = False )
df3.to_csv("Plots/ANNDerivRAWLag"+str(Lag1)+"_Ratio.csv", index = False)
