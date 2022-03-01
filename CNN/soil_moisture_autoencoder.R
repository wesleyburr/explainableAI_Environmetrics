rm=list(ls())

library(readr)
library(lubridate)
library(dplyr)
library(ggplot2)
library(keras)


### Reading Soil Moisture Data ####
cols <- paste0( "t",
                rep(1948:2021, each = 12),
                "M",
                rep(c("01", "02", "03", "04", "05", "06", 
                      "07", "08", "09", "10", "11", "12"), 2021-1948+1) )

Soil_original <- read.csv('Soil.csv')

Soil_original <- Soil_original %>% filter(between(Lon,-101.5,-80.5)&between(Lat,35.5,48.5))

SoilMoisture_lonlat <- Soil_original[,1:2]
names(SoilMoisture_lonlat) <- c("LON", "LAT")

# data
SoilMoisture <- Soil_original[,3:ncol(Soil_original)]
names(SoilMoisture) <- cols[1:ncol(SoilMoisture)]

Soil_lon <- sort(unique(SoilMoisture_lonlat$LON))
Soil_lat <- sort(unique(SoilMoisture_lonlat$LAT))

plot(SoilMoisture_lonlat$LON,SoilMoisture_lonlat$LAT)

nlon_soil <- length(Soil_lon)
nlat_soil <- length(Soil_lat)
ntime_soil <- ncol(SoilMoisture)


### Reading SST Data ####
SST_original <- read_csv("SST_data.csv")

SST_lonlat = SST_original[,2:3]
names(SST_lonlat) <- c("LON", "LAT")

SST_lon=sort(unique(SST_lonlat$LON))
SST_lat=sort(unique(SST_lonlat$LAT))

nlon <- length(SST_lon)
nlat <- length(SST_lat)

# data
SST_anom <- SST_original[,4:ncol(SST_original)]
names(SST_anom) <- cols[1:ncol(SST_anom)]

ntime <- ncol(SST_anom)



#Creating 4D arrays to be used in CNN layers

# SST_anom_array <- array(NA,dim=c(nlon,nlat,ntime))
# 
# for(i in 1:nlon)
# {
#   for(j in 1:nlat)
#   {
#     location_index <- SST_lonlat$LON==SST_lon[i]&SST_lonlat$LAT==SST_lat[j]
#     for(k in 1:ntime)
#     {
#       if(sum(location_index)==1) SST_anom_array[i,j,k] <- unlist(SST_anom[location_index,k])
#       if(sum(location_index)==0) SST_anom_array[i,j,k] <- NA
#       if(sum(location_index)>1)
#       {
#         print('error!');break
#       }
#     }
#   }
# }
# 
# saveRDS(file='SST_anom_array.rds',SST_anom_array)

SST_anom_array <- readRDS(file='SST_anom_array.rds')
#Check the resulting data 
image(SST_lon,SST_lat,SST_anom_array[,,1])



SST_anom_train <- array(NA,dim=c(789,nlon,nlat,1))

for(t in 1:789)
{
  SST_anom_train[t,,,1]=SST_anom_array[,,t]
}

SST_anom_train=SST_anom_train[,,2:45,]

SST_anom_train[is.na(SST_anom_train)]=0


SST_anom_test <- array(NA,dim=c(885-792+1,nlon,nlat,1))

for(t in 792:885)
{
  SST_anom_test[t-791,,,1]=SST_anom_array[,,t]
}

SST_anom_test=SST_anom_test[,,2:45,]

SST_anom_test[is.na(SST_anom_test)]=0


#building an autoencoder
enc_input = layer_input(shape = c(84*44))

enc_output = enc_input %>% 
  layer_reshape(target_shape=c(84,44,1)) %>%
  layer_conv_2d(3,kernel_size=c(3,3), activation="relu", padding="same") %>% 
  layer_max_pooling_2d(c(2,2), padding="same") %>% 
  layer_conv_2d(1,kernel_size=c(3,3), activation="relu", padding="same") %>% 
  layer_max_pooling_2d(c(4,4), padding="same")  

dec_output = enc_output %>% 
  layer_conv_2d(1, kernel_size=c(3,3), activation="relu", padding="same") %>% 
  layer_upsampling_2d(c(4,4)) %>% 
  layer_conv_2d(3, kernel_size=c(3,3), activation="relu") %>% 
  layer_upsampling_2d(c(2,2)) %>% 
  layer_conv_2d(1, kernel_size=c(3,3), activation="linear", padding="same")

aen = keras_model(enc_input, dec_output)

aen %>% compile(optimizer="adam", loss="mean_squared_error")

summary(aen)

SST_anom_train_mat <- array_reshape(SST_anom_train,c(nrow(SST_anom_train),84*44))

SST_anom_test_mat <- array_reshape(SST_anom_test,c(nrow(SST_anom_test),84*44))

aen %>% fit(SST_anom_train_mat, SST_anom_train, epochs=400, batch_size=200)


#Response Variable
Y_train <- t(as.matrix(SoilMoisture[,4:792]))

Y_test <- t(as.matrix(SoilMoisture[,795:888]))

Y_test <- (Y_test-min(Y_train))/(max(Y_train)-min(Y_train))

Y_train <- (Y_train-min(Y_train))/(max(Y_train)-min(Y_train))


#Create a model for predicting the response variable

conv1 <- enc_output %>%
  layer_flatten() 

concatnated_layer <- conv1

prediction  <- concatnated_layer %>%
  layer_dense(units=100) %>%
  layer_activation('relu') %>% 
  layer_dense(units=100) %>%
  layer_activation('relu') %>%
  layer_dense(units=100,activation="relu")  %>%
  layer_dense( units = ncol(Y_train) , activation = "linear" )  

model <- keras_model( enc_input, prediction)

summary(model)

model %>% compile(
  optimizer ='adam',
  loss = 'mean_squared_error',
  metrics = 'mse',
)

history <- model %>% fit(
  SST_anom_train_mat, Y_train,
  epochs = 300, batch_size = 100, learning_rate=0.000001
)

#Checking the model performance

cor(c(model$predict(list(SST_anom_train_mat))),c(Y_train))

plot(c(model$predict(list(SST_anom_test_mat)))[seq(1,422004,,1000)],c(Y_test)[seq(1,422004,,1000)])

