library(tidyverse)
library(fields)
library(broom)
library(gridExtra)
library(glmnet)
library(lubridate)
library(modelr)
theme_set(theme_classic(base_size = 25))

SoilMoisture.all <- read.csv("../Soil.csv", header = T) %>%
  tibble::rowid_to_column("sm_loc_id")

### function that inputs a model name and 
### outputs table of metrics (RMSE, R2, skill), 
### scatter plot of fits for that model, and MSPE by location
### for all hold-out data (Jan 2014 onwards) and 
### also only for the month of May
metrics_gen=function(model_name){
  
  tab.fit=read.csv(paste0("outputs/",model_name,"_fits.csv"))
  
  ## MSPE and R2
  tab.MSPE=tab.fit %>%
  dplyr::filter(date > date.in) %>%
  transmute(error = (value-fit)^2) %>%
  pull %>% mean
  
  tab.R2=tab.fit %>%
  dplyr::filter(date > date.in) %>%
  summarise(R2=cor(value,fit)^2) %>% pull
  
  ## metrics only for May
  tab.MSPE.May=tab.fit %>%
  dplyr::filter(date > date.in,month(date)==5) %>%
  transmute(error = (value-fit)^2) %>%
  pull %>% mean
  
  tab.R2.May=tab.fit %>%
  dplyr::filter(date > date.in,month(date)==5) %>%
  summarise(R2=cor(value,fit)^2) %>% pull
  
  ### skill score using persistence fits as baseline
  pers.fit=read.csv(paste0("outputs/persistence_fits.csv"))
  
  pers.MSPE=pers.fit %>%
  dplyr::filter(date > date.in) %>%
  transmute(error = (value-fit)^2) %>%
  pull %>% mean
  
  pers.MSPE.May=pers.fit %>%
  dplyr::filter(date > date.in,month(date)==5) %>%
  transmute(error = (value-fit)^2) %>%
  pull %>% mean
  
  tab.metrics=data.frame(period=c("ALL months","May"),
  MSPE=c(tab.MSPE,tab.MSPE.May),R2=c(tab.R2,tab.R2.May)
  ,skill=c(1 - tab.MSPE/pers.MSPE,1 - tab.MSPE.May/pers.MSPE.May))

  write.csv(tab.metrics,paste0("metrics/",model_name,"_metrics.csv"),row.names = F,quote = F)
  
  ## scatterplot of fits
  png(paste0("figures/",model_name,"_fits.png"),width = 480, height = 360)
  p=ggplot(tab.fit,aes(x=value,y=fit,col=year(date))) +
  geom_point(alpha=0.5) +
  scale_color_viridis_c(direction=-1) +
  geom_abline(slope=1) +
  labs(col="year",y= "fit (in mm.)", x = "true value (in mm.)")
  print(p)
  dev.off()
  
  ## scatterplot of fits for May
  png(paste0("figures/",model_name,"_fits_May.png"),width = 480, height = 360)
  p=ggplot(tab.fit %>% dplyr::filter(month(date)==5),aes(x=value,y=fit,col=year(date))) +
  geom_point(alpha=0.5) +
  scale_color_viridis_c(direction=-1) +
  geom_abline(slope=1) +
  labs(col="year",y= "fit (in mm.)", x = "true value (in mm.)")
  print(p)
  dev.off()

  ##### plot MSPE (all months and May) by corn belt location

  tab.fit.byloc=tab.fit %>% 
  select(sm_loc_id,Lon,Lat,value,fit) %>%
  group_by(sm_loc_id) %>%
  summarize(Lon=first(Lon),Lat=first(Lat),
            MSPE=mean((value-fit)^2))

  tab.fit.byloc.all = SoilMoisture.all %>% 
    select(sm_loc_id,Lon,Lat) %>%
    left_join(tab.fit.byloc) %>%
    mutate(months="All months")
  
  tab.fit.byloc.May=tab.fit %>% 
  dplyr::filter(month(date)==5) %>%
  select(sm_loc_id,Lon,Lat,value,fit) %>%
  group_by(sm_loc_id) %>%
  summarize(Lon=first(Lon),Lat=first(Lat),
            MSPE=mean((value-fit)^2))
  
  tab.fit.byloc.all.May = SoilMoisture.all %>% 
  select(sm_loc_id,Lon,Lat) %>%
  left_join(tab.fit.byloc.May) %>%
  mutate(months="May")
  
  png(paste0("figures/",model_name,"_mspe_byloc.png"),width=960,height=360)
  p=rbind(tab.fit.byloc.all,tab.fit.byloc.all.May) %>%  
    ggplot() +
    geom_point(aes(x=Lon,y=Lat,col=MSPE)) +
    scale_color_viridis_c(direction=1) +
    facet_wrap(. ~ months)
  print(p)
  dev.off()
}

# metrics_gen("persistence")
# metrics_gen("climatology")
# metrics_gen("funclm")

