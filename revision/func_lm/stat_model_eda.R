library(tidyverse)
library(fields)
library(broom)
library(gridExtra)
library(glmnet)
library(lubridate)
theme_set(theme_classic(base_size = 25))

#### soil moisture data

SoilMoisture.all <- read.csv("../Soil.csv", header = T) %>%
  tibble::rowid_to_column("sm_loc_id")

# long format
SoilMoisture.long=pivot_longer(SoilMoisture.all,cols=-c(Lon,Lat,sm_loc_id),names_to="date") %>% 
  mutate(date=as.POSIXct(strptime(gsub("X","",date),"%m.%d.%Y")))

#write.csv(SoilMoisture.long,"../SoilMoisturelong.csv",quotes=F,row.names=F)

#### SST data
lag=3

SST.all = read.csv("../SST_data.csv", header = T) %>%
  mutate(sst_loc_id=X) %>%
  select(-X)

# long-format plus 3 months time-lag to predict the soil moisture data

### adding lag months to the date for SST data (for pairing with SM data)

SST.long=pivot_longer(SST.all,cols=-c(Lon,Lat,sst_loc_id),names_to="date") %>% 
  mutate(date=as.POSIXct(strptime(gsub("X","",date),"%Y.%m.%d"))) %>%
  mutate(lagged.date=date %m+% months(lag))

SST.lagged=SST.long %>%
  mutate(origdate=date) %>%
  mutate(date=lagged.date) %>%
  select(-lagged.date)

### regression near the corn belt (35.5 - 48.5 N, -101.5 - -80.5 W)

corn.belt.index=which(SoilMoisture.all$Lat >=35.5 & 
                      SoilMoisture.all$Lat <=48.5 &
                      SoilMoisture.all$Lon >=-101.5 &
                      SoilMoisture.all$Lon <=-80.5)

length(corn.belt.index)

png("figures/cornbelt.png",width = 720, height = 480)
plot(SoilMoisture.all[,c("Lon","Lat")],cex.lab=2, cex.axis=2)
points(SoilMoisture.all[corn.belt.index,c("Lon","Lat")],col="red",pch=16)
dev.off()

SoilMoisture.corn.belt.all = SoilMoisture.long %>%
  dplyr::filter(Lat >=35.5,Lat <=48.5,Lon >=-101.5,Lon <=-80.5) 
  
SoilMoisture.corn.belt.wide = SoilMoisture.corn.belt.all %>%
  select(-c(Lon,Lat)) %>%
  pivot_wider(id_cols=c(date),
  names_from=sm_loc_id,values_from=value,
  names_prefix="sm")

SoilMoisture.corn.belt.mean = cbind(SoilMoisture.corn.belt.wide[,1],
  rowMeans(SoilMoisture.corn.belt.wide[,-1]))
colnames(SoilMoisture.corn.belt.mean)=c("date","mean.SoilMoisture")

SST.with.sm = SST.lagged %>% 
  left_join(SoilMoisture.corn.belt.mean) %>%
  dplyr::filter(!is.na(mean.SoilMoisture))

SST.corr=SST.with.sm %>% 
  group_by(Lon,Lat,sst_loc_id) %>%
  summarize(corr=cor(value,mean.SoilMoisture))

p=ggplot(data=SST.corr,aes(x=Lon,y=Lat,col=corr)) +
  geom_point(size=2.2) +
  #scale_fill_viridis_c() +
  scale_color_gradient2(low = "blue", mid = "white",
                            high = "red", space = "Lab") +
  geom_hline(yintercept=c(0))

png("figures/sst_corr.png",width = 720, height = 480)
print(p)
dev.off()
