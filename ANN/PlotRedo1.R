# Set the working directory
# setwd("/Volumes/GoogleDrive/.shortcut-targets-by-id/1ZaD4u22lYjzvmqq5twsPvWU3ecPu4pkq/Working Group /SoilMoistureExample/Model_ANN")
#setwd("G:/.shortcut-targets-by-id/1ZaD4u22lYjzvmqq5twsPvWU3ecPu4pkq/Working Group/SoilMoistureExample/Model_ANN")
setwd("C:/Users/Ed/Documents/GitHub/explainableAI_Environmetrics/ANN")

library(tidyverse)
library(fields)
library(broom)
library(gridExtra)
library(glmnet)
library(lubridate)
library(modelr)
theme_set(theme_classic(base_size = 25))


x1 <- read.csv( "plots/ANNDerivWideRatioRaw.csv")

d1 <- data.frame(Lon = x1$X, Lat = x1$Y, MSPEratio = x1$Z4, months = "January" )

png("figures/PredDec_ratio.png",width=960,height=360)
p = d1 %>%  
  ggplot() +
  geom_point(aes(x=Lon,y=Lat,col=MSPEratio)) +
  scale_color_viridis_c(direction=1) +
  ggtitle(label = expression(paste("February to May (RAW Wide)")),
          subtitle = "December")
#facet_wrap(. ~ months)
print(p)
dev.off()







x1 <- read.csv( "plots/ANNDerivWideRatioPCA.csv")

d1 <- data.frame(Lon = x1$X, Lat = x1$Y, MSPEratio = x1$Z2, months = "January" )

png("figures/Pred_Oct_to_May_PCA_ratio.png",width=960,height=360)
p = d1 %>%  
  ggplot() +
  geom_point(aes(x=Lon,y=Lat,col=MSPEratio)) +
  scale_color_viridis_c(direction=1) +
  ggtitle(label = expression(paste("February to May (", PCA[60]," Wide)")),
   subtitle = "October")
  #facet_wrap(. ~ months)
print(p)
dev.off()

##########################################################
# Individual Month Plots
# For Raw 2014
x1 <- read.csv( "Plots/ANNDerivRAWFeb_to_May_Ratio2014.csv")
d1 <- data.frame(Lon = x1$X, Lat = x1$Y, MSPEratio = x1$Z0, months = "January" )
png("figures/Pred_Feb_to_May_Ratio2014.png",width=960,height=360)
p = d1 %>%  
  ggplot() +
  geom_point(aes(x=Lon,y=Lat,col=MSPEratio)) +
  scale_color_viridis_c(direction=1) +
  ggtitle(label = expression(paste("2014 May (RAW) ", R[pred]^{2}," = 0.608")))
#facet_wrap(. ~ months)
print(p)
dev.off()

# For Raw 2015
x1 <- read.csv( "Plots/ANNDerivRAWFeb_to_May_Ratio2015.csv")
d1 <- data.frame(Lon = x1$X, Lat = x1$Y, MSPEratio = x1$Z0, months = "January" )
png("figures/Pred_Feb_to_May_Ratio2015.png",width=960,height=360)
p = d1 %>%  
  ggplot() +
  geom_point(aes(x=Lon,y=Lat,col=MSPEratio)) +
  scale_color_viridis_c(direction=1) +
  ggtitle(label = expression(paste("2015 May (RAW) ", R[pred]^{2}," = 0.364")))
#facet_wrap(. ~ months)
print(p)
dev.off()

# For Raw 2016
x1 <- read.csv( "Plots/ANNDerivRAWFeb_to_May_Ratio2016.csv")
d1 <- data.frame(Lon = x1$X, Lat = x1$Y, MSPEratio = x1$Z0, months = "January" )
png("figures/Pred_Feb_to_May_Ratio2016.png",width=960,height=360)
p = d1 %>%  
  ggplot() +
  geom_point(aes(x=Lon,y=Lat,col=MSPEratio)) +
  scale_color_viridis_c(direction=1) +
  ggtitle(label = expression(paste("2016 May (RAW) ", R[pred]^{2}," = 0.431")))
#facet_wrap(. ~ months)
print(p)
dev.off()

##########################################################
# Individual Month Plots
# For PCA60 2014
x1 <- read.csv( "Plots/ANNDeriv_Feb_to_May_Ratio2014.csv")
d1 <- data.frame(Lon = x1$X, Lat = x1$Y, MSPEratio = x1$Z0, months = "January" )
png("figures/Pred_PCA_Feb_to_May_Ratio2014.png",width=960,height=360)
p = d1 %>%  
  ggplot() +
  geom_point(aes(x=Lon,y=Lat,col=MSPEratio)) +
  scale_color_viridis_c(direction=1) +
  ggtitle(label = expression(paste("2014 May (",PCA[60],") ", R[pred]^{2}," = 0.538")))
#facet_wrap(. ~ months)
print(p)
dev.off()

# For PCA60 2015
x1 <- read.csv( "Plots/ANNDeriv_Feb_to_May_Ratio2015.csv")
d1 <- data.frame(Lon = x1$X, Lat = x1$Y, MSPEratio = x1$Z0, months = "January" )
png("figures/Pred_PCA_Feb_to_May_Ratio2015.png",width=960,height=360)
p = d1 %>%  
  ggplot() +
  geom_point(aes(x=Lon,y=Lat,col=MSPEratio)) +
  scale_color_viridis_c(direction=1) +
  ggtitle(label = expression(paste("2015 May (",PCA[60],") ", R[pred]^{2}," = 0.468")))
#facet_wrap(. ~ months)
print(p)
dev.off()

# For PCA60 2016
x1 <- read.csv( "Plots/ANNDeriv_Feb_to_May_Ratio2016.csv")
d1 <- data.frame(Lon = x1$X, Lat = x1$Y, MSPEratio = x1$Z0, months = "January" )
png("figures/Pred_PCA_Feb_to_May_Ratio2016.png",width=960,height=360)
p = d1 %>%  
  ggplot() +
  geom_point(aes(x=Lon,y=Lat,col=MSPEratio)) +
  scale_color_viridis_c(direction=1) +
  ggtitle(label = expression(paste("2016 May (",PCA[60],") ", R[pred]^{2}," = 0.329")))
#facet_wrap(. ~ months)
print(p)
dev.off()

