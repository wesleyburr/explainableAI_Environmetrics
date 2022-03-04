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
from random import sample
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats as sc

# Change the working directory
# os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN")
# os.chdir("/home/ed/Documents/TiesWG/Model_ANN")
os.chdir("/home/ed/Documents/GitHub/explainableAI_Environmetrics/ANN")

# Set the desired lag...
Lag1 = 3

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
SoilTest = SoilMoist1[:,797]
SoilTesta = SoilMoist1a[:,797]

# Hold out for Shuffle
SSThold1 = SSTanom2[:,780:890]

pca_top1 = PCA( n_components = 60 )
pca_top1.fit( SSTTrain) 
STTrain_pca = pca_top1.transform( SSTTrain )
 
plt.grid()
plt.plot( np.cumsum( pca_top1.explained_variance_ratio_ * 100 ) ) 


# Build a model
model = Sequential()
model.add( Dense( 100, input_dim = 60, activation = 'tanh' ) )
model.add( Dense( 100, activation = 'tanh' ) )
model.add( Dense( 500, activation = 'tanh' ) )
model.add( Dense( 1224 ) )#, activation = 'tanh' ) )

# # Compile the model
model.compile( loss = 'mean_squared_error', optimizer='adam')
    
# # fit the keras model on the dataset
model.fit(STTrain_pca, SoilTrain, epochs=500)#, batch_size=10)


SSTtest = pca_top1.transform( np.reshape( SSTanom2[:,794], (1, 3186) ) )
SoilTest = SoilMoist1[:,797]

# Predict with the model
y1 = model.predict( SSTtest )
y1a = model.predict( SSTtest )*SoilMoist1s[797] + SoilMoist1bm[797] 
MSE1 = MeanSquaredError()
y1MSE = MSE1( SoilTest, y1 ).numpy()
y1MSEa = MSE1( SoilTesta, y1a ).numpy()
plt.figure()
plt.plot( SoilTest, y1.T ,'o')
predR2 = sc.pearsonr(y1.T[:,0], SoilTest)[0]**2






# Data for a three-dimensional line
zpoints1 = SoilTrain[0,:]
xpoints1 = LandData1['Lat']
ypoints1  = LandData1['Lon']


# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilMoist1[:,797]
SSTFirst = SSTanom2[:,(794)]
Zsens1 = np.copy( SSTFirst )
Zn1 = np.shape(Zsens1)[0]
Zsens2 = np.copy( SSTFirst )
Zsens3 = np.zeros( ( 3186, 20 ) )
for i in range( 0, Zn1 ):
   Ztest = np.copy( SSTFirst )
   for j in range(0, 20):
       index1 = sample( list(range(0,90)) ,20)
       Ztest[i] = SSThold1[ i, index1[j]  ]
       Zp = model.predict( pca_top1.transform( Ztest.reshape([1,Zn1] ) ) )
       Zsens3[i,j] = MSE1( Z1, Zp ).numpy()


# Pack it into a data frame
data1 = [ zpoints1, xpoints1, ypoints1 ]
df= pd.DataFrame( data = np.transpose(data1), columns = ['Z','X','Y'] )

#Zsens2mx = np.max( Zsens2 )
#Zsens2mn = np.min( Zsens2 )
#Zsens1s = (1-(Zsens2 - Zsens2mn)/(Zsens2mx - Zsens2mn))
#Zsens2s = np.reshape( Zsens1s, (6,3186))
#data2 = np.hstack( (SSTlonlat, np.reshape(Zsens1s, (3186,1))))
#df2 = pd.DataFrame( data2 , columns=['X','Y','Z0'] )
#Zsens3mx = np.max( Zsens3 )
#Zsens3mn = np.min( Zsens3 )
#Zsens3s = 1 - (Zsens3 - Zsens3mn)/(Zsens3mx - Zsens3mn)
Zsens3s1 = np.mean( Zsens3, axis = 1)-y1MSE
data3 =  np.hstack( ( SSTlonlat , np.reshape(Zsens3s1, (3186,1) )))
df3 = pd.DataFrame( data3,  columns=['X','Y','Z0'] )
# Write the data out...
df3.to_csv("Plots/ANNDeriv_Feb_to_May_Ratio2014Shuffle.csv", index = False)
# df3 = pd.read_csv("Plots/ANNDeriv_Feb_to_May_Ratio2014.csv")

#################################################################################
#  Make some pictures
#

fig = plt.figure()
plt.scatter( df3.X, df3.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, 
            #s = 5*df3.Z0, 
            s = 3.7,
            c = df3.Z0, cmap = 'bwr')#, vmin = 1- (1-1.0081), vmax = 1.0081 ) #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('Lon')
plt.ylabel('Lat')
plt.title('2014 May (PCA$_{60}$) Feature Shuffle')
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_PCA_Feb_to_May_Ratio2014Shuffle.png', format = 'png')#, quality = 100)
plt.show()


