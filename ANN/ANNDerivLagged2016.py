import os
import numpy as np
import pandas as pd
#from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
#import tensorflow as tf
#tf.config.run_functions_eagerly(True)
#from tensorflow.keras.layers import Conv2D
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
# from sklearn.preprocessing import scale
#from sklearn.pipeline import Pipeline
#from mpl_toolkits import mplot3d
#import seaborn as sb
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats as sc

# Change the working directory
# os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN")
# os.chdir("/home/ed/Documents/TiesWG/Model_ANN")
# os.chdir("G:/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample/Model_ANN")
# os.chdir("/Users/ed/Documents/GitHub/Spatial/explainableAI_Environmetrics/ANN")
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
LandData1 = SoilMoist1in[ (['Unnamed: 0','Lon','Lat']) ];
LandData1.rename(columns = {'Unnamed: 0':'sm_loc_id'});
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
SoilTrain =  np.transpose( SoilMoist1[:,Lag1:(792+Lag1+24)] )
SSTTrain = np.transpose( SSTanom2[:,0:(792+24)] )
SSTFirst = SSTTrain[0,:]
SSTTrainSd1 = np.std(SSTTrain, axis = 0 )
SoilTest = SoilMoist1[:,(796+24)]
SoilTesta = SoilMoist1a[:,(796+24)]


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
model.add( Dense( 1224 ) )#, activation = 'tanh' ) )

# # Compile the model
model.compile( loss = 'mean_squared_error', optimizer='adam' )
    
# # fit the keras model on the dataset
model.fit(SSTTrain, SoilTrain, epochs=200)#, batch_size=10)


SSTtest = np.reshape( SSTanom2[:,(793+24)], (1, 3186) ) 
SoilTest = SoilMoist1[:,(796+24)]

# Predict with the model
y1a = model.predict( SSTtest )*SoilMoist1s[(796+24)] + SoilMoist1bm[(796+24)] 
y1 = model.predict( SSTtest ) 
#MSE1 = MeanSquaredError()
#y1MSE = MSE1( SoilTest, y1 ).numpy()
#y1MSEa = MSE1( SoilTesta, y1a ).numpy()
plt.plot( SoilTest, y1.T ,'o')
predR2 = sc.pearsonr(y1.T[:,0], SoilTest)[0]**2


# Get out the predicted values
y1b = pd.DataFrame( y1a.T, columns=['fit'])
y1c = pd.DataFrame( SoilMoist1b[:,(796+24)], columns = ['value'] )
date1 = pd.DataFrame( pd.Series( np.tile(['5/1/2016'], 1224) ), columns =['date'] )
LandData2 = LandData1[ (['Lon','Lat']) ]
LD1 = pd.DataFrame( np.asarray(LandData1['Unnamed: 0']), columns = ['sm_loc_id'] )
data3 =  pd.concat( [LD1, LandData2, date1, y1c, y1b], axis = 1 )
data3.to_csv('outputs/ANN2016_pred.csv', index = False )


# Get out the fitted values
SSTTrain_full = np.transpose( SSTanom2[:,0:(884)] )
y1_fit = model.predict( SSTTrain_full )
y1_fita = y1_fit[0,:]*SoilMoist1s[(0)] + SoilMoist1bm[(0)] 
col1 = list( SoilMoist1in.columns )[(3+Lag1):891]
col2 = [s.replace("X", "") for s in col1 ]
date2 = pd.DataFrame(pd.Series([s.replace(".", "/") for s in col2 ]), columns = ['date'])
date3 = pd.DataFrame( pd.Series( np.tile(date2.loc[0], 1224) ), columns =['date'] )
val1 = pd.DataFrame(  SoilMoist1b[:,0], columns = ['value'] )
y1_fit2 = pd.DataFrame( y1_fita.T, columns = ['fit'] )
fit1 = pd.concat( [LD1, LandData2, date3, val1, y1_fit2] , axis = 1 )
for i in (n+1 for n in range(883) ):
    y1_fita = y1_fit[i,:]*SoilMoist1s[(i)] + SoilMoist1bm[(i)] 
    date3 = pd.DataFrame( pd.Series( np.tile(date2.loc[i], 1224) ), columns =['date'] )
    val1 = pd.DataFrame(  SoilMoist1b[:,i], columns = ['value'] )
    y1_fit2 = pd.DataFrame( y1_fita.T, columns = ['fit'] )
    fit1a = pd.concat( [LD1, LandData2, date3, val1, y1_fit2] , axis = 1 )
    fit1 = pd.concat( [fit1, fit1a], axis = 0 )
    
 

fit1.to_csv('outputs/ANN2016_fits.csv', index = False )




y1b = pd.DataFrame( y1a.T, columns=['fit'])
y1c = pd.DataFrame( SoilTesta.T, columns = ['value'] )
date1 = pd.Series( np.tile(['5/1/2016'], 1224) )
data3 =  pd.concat( [LandData1, date1, y1c, y1b], axis = 1 )
data3.to_csv('outputs/ANN2016_fits.csv', index = False )


# Data for a three-dimensional line
zpoints1 = SoilTrain[0,:]
xpoints1 = LandData1['Lat']
ypoints1  = LandData1['Lon']

# Predict the data...
Zsd1 = np.std( SoilTrain[0,:] )
Z1 = SoilMoist1[:,(797+24)]
SSTFirst = SSTanom2[:,(794+24)]
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
data3 =  np.hstack( ( SSTlonlat , np.reshape(Zsens3 , (3186,1) )))
df3 = pd.DataFrame( data3,  columns=['X','Y','Z0'] )


# Write the data out...
df3.to_csv("Plots/ANNDerivRAWFeb_to_May_Ratio2016.csv", index = False)


#################################################################################
#  Make some pictures
#

fig = plt.figure()
plt.scatter( df2.X, df2.Y, s = 0.1, c = 'white')
plt.scatter( df3.X, df3.Y, s = 5*df3.Z0, c = df3.Z0, cmap = 'bwr') #, alpha = 0.5)
plt.hlines( 0, xmin = np.min(df3.X), xmax = np.max(df3.X), linewidth = 0.5, colors = "black")
plt.colorbar()
plt.xlabel('Lon')
plt.ylabel('Lat')
plt.title('2016 May (RAW) $R^2_{pred}=$'+str(np.round(predR2,3)))
#os.chdir("/Volumes/GoogleDrive/My Drive/Research/Working Group /SoilMoistureExample")
plt.savefig('Plots/Pred_Feb_to_May_Ratio2016.png', format = 'png')#, quality = 100)
plt.show()




