library(tidyverse)
library(fields)
library(broom)
library(gridExtra)
library(glmnet)
library(lubridate)
library(modelr)

#### soil moisture data

SoilMoisture.all <- read.csv("../Soil.csv", header = T) %>%
  tibble::rowid_to_column("sm_loc_id")

# long format
SoilMoisture.long=pivot_longer(SoilMoisture.all,cols=-c(Lon,Lat,sm_loc_id),names_to="date") %>% 
  mutate(date=as.POSIXct(strptime(gsub("X","",date),"%m.%d.%Y")))

## adding the 3-month lagged term for autocorrelation
lag=3

SoilMoisture.lagged=SoilMoisture.long %>%
  transmute(sm_loc_id,lagged=value,date=date %m+% months(lag))

SoilMoisture.long = SoilMoisture.long %>%
  left_join(SoilMoisture.lagged) %>%
  dplyr::filter(!is.na(lagged))

### regression near the corn belt (35.5 - 48.5 N, -101.5 - -80.5 W)
SoilMoisture.corn.belt.all = SoilMoisture.long %>%
  dplyr::filter(Lat >=35.5,Lat <=48.5,Lon >=-101.5,Lon <=-80.5) 

### persistence fits

### in-sample period
date.in = ymd("2013-12-31")

pers.MSPE=SoilMoisture.corn.belt.all %>%
  dplyr::filter(date > date.in) %>%
  transmute(error = (value-lagged)^2) %>%
  pull %>% mean

pers.R2=SoilMoisture.corn.belt.all %>%
  dplyr::filter(date > date.in) %>%
  summarise(R2=cor(value,lagged)^2) %>% pull

## fits only for May
pers.MSPE.May=SoilMoisture.corn.belt.all %>%
  dplyr::filter(date > date.in,month(date)==5) %>%
  transmute(error = (value-lagged)^2) %>%
  pull %>% mean

pers.R2.May=SoilMoisture.corn.belt.all %>%
  dplyr::filter(date > date.in,month(date)==5) %>%
  summarise(R2=cor(value,lagged)^2) %>% pull

pers.table=data.frame(period=c("ALL months","May"),
  MSPE=c(pers.MSPE,pers.MSPE.May),R2=c(pers.R2,pers.R2.May))

#### SST data
SST.all = read.csv("../SST_data.csv", header = T) %>%
  mutate(sst_loc_id=X) %>%
  select(-X)

### pca
X=scale(as.matrix(SST.all %>% select(starts_with("X"))))
colnames(X)=NULL
row.names(X)=NULL

pca=prcomp(X,scale=F,center = F)

rank=first(which(cumsum(pca$sdev^2)/sum(pca$sdev^2) > 0.8))
rank

basis=(pca$x)[,1:rank]
colnames(basis)=paste0("phi",1:rank)

pca.df=cbind(SST.all[,c("Lon","Lat")],basis)

pca.df %>% 
  pivot_longer(cols=-c(Lat,Lon),names_to="pca") %>%
  mutate(pca=as.numeric(gsub("phi","",pca))) %>%
  dplyr::filter(pca <= 12) %>%
  ggplot(aes(x=Lon,y=Lat,col=value)) + 
  geom_point() +
  facet_wrap(.~pca, scales="free") +
  scale_color_viridis_c(direction = -1,alpha=0.5) +
  geom_hline(yintercept=0)

### functional regression model with pca basis
#### generating basis for the SST_locations

SST.bas = as.data.frame(basis)
colnames(SST.bas)=paste0("V",1:rank)

### creating the inner products of pca basis with the SST data
inner.mat = t(X)%*%as.matrix(SST.bas)/nrow(SST.all)
row.names(inner.mat)=NULL

## final matrix of covariates with time shifted by 3 months (for 3 months ahead prediction)
SST.lagged.dates=as.POSIXct(strptime(gsub("X","",
  colnames(SST.all %>% select(starts_with("X")))),"%Y.%m.%d")) %m+% months(lag)
SST.inner = cbind(data.frame(date=SST.lagged.dates),as.data.frame(inner.mat))

SoilMoisture.corn.belt = SoilMoisture.corn.belt.all %>%
 left_join(SST.inner) %>%
 mutate(month=month(date))

SoilMoisture.corn.belt.in=SoilMoisture.corn.belt  %>% 
  dplyr::filter(date <= date.in)

SoilMoisture.corn.belt.out=SoilMoisture.corn.belt %>% 
  dplyr::filter(date > date.in)

lm.formula = paste0("value ~ lagged + as.factor(month) + ", paste(paste0("V",1:ncol(inner.mat)),collapse=" + "))

### regression ###
SM.corn.belt.output=SoilMoisture.corn.belt.in %>% 
   group_by(sm_loc_id) %>%
   do(fitSM = lm(lm.formula, data = .))

SoilMoisture.corn.belt.out.pred=SoilMoisture.corn.belt.out %>% 
  group_by(sm_loc_id) %>%
  nest %>% 
  inner_join(SM.corn.belt.output) %>% 
  mutate(pred = map2(fitSM, data, predict)) %>% 
  unnest(c(data,pred)) %>%
  as.data.frame %>%
  select(sm_loc_id,Lon,Lat,date,month,value,lagged,pred)

ggplot(SoilMoisture.corn.belt.out.pred,aes(x=value,y=pred,col=year(date))) +
  geom_point(alpha=0.2) +
  scale_color_viridis_c(direction=-1) +
  geom_abline(slope=1) +
  labs(col="year")

ggplot(SoilMoisture.corn.belt.out.pred %>% dplyr::filter(month==5),
       aes(x=value,y=pred,col=year(date))) +
  geom_point(alpha=0.2) +
  scale_color_viridis_c(direction=-1) +
  geom_abline(slope=1) +
  labs(col="year")

MSPE=mean((SoilMoisture.corn.belt.out.pred$pred-SoilMoisture.corn.belt.out.pred$value)^2)
R2 = cor(SoilMoisture.corn.belt.out.pred$pred,SoilMoisture.corn.belt.out.pred$value)^2
c(MSPE,R2)

skill.score = 1 - MSPE/pers.MSPE
skill.score 

SoilMoisture.corn.belt.out.May = SoilMoisture.corn.belt.out.pred %>% 
  dplyr::filter(month==5)

MSPE.May=mean((SoilMoisture.corn.belt.out.May$pred-SoilMoisture.corn.belt.out.May$value)^2)
R2.May = cor(SoilMoisture.corn.belt.out.May$pred,SoilMoisture.corn.belt.out.May$value)^2
c(MSPE.May,R2.May)

skill.score.May = 1 - MSPE.May/pers.MSPE.May
skill.score.May 

fits.table=data.frame(period=c("All months","May"),
  MSPE=c(MSPE,MSPE.May),R2=c(R2,R2.May),skill=c(skill.score,skill.score.May))

### plotting the coefficient functional surface
coeff=Reduce('+',lapply(SM.corn.belt.output$fitSM,function(x) x$coefficients))/length(unique(SM.corn.belt.output$sm_loc_id))
surface=cbind(SST.all[,c("Lon","Lat")],data.frame(coeff=as.matrix(SST.bas)%*%tail(coeff,rank)))

ggplot(data=SST.all,aes(x=Lon,y=Lat)) +
  geom_point() +
geom_point(data=surface,aes(x=Lon,y=Lat,col=coeff),size=1.2) +
  scale_color_gradient2(low = "blue", mid = "white",
                            high = "red", space = "Lab" ) +
  geom_hline(yintercept=c(0)) +
  theme_classic()

##### plot MSPE and skill by corn belt location

SoilMoisture.corn.belt.pred=SoilMoisture.corn.belt.out.pred %>% 
  select(sm_loc_id,Lon,Lat,value,lagged,pred) %>%
  group_by(sm_loc_id) %>%
  summarize(Lon=first(Lon),Lat=first(Lat),
            MSPE=mean((value-pred)^2))

SoilMoisture.all %>% 
ggplot(aes(x=Lon,y=Lat)) +
  geom_point(shape=1) +
  geom_point(data=SoilMoisture.corn.belt.pred,
             aes(x=Lon,y=Lat,col=MSPE)) +
  scale_color_viridis_c(direction=1)

  
##### plot MSPE and skill by corn belt location for May
SoilMoisture.corn.belt.pred.May=SoilMoisture.corn.belt.out.pred %>% 
  dplyr::filter(month==5) %>%
  select(sm_loc_id,Lon,Lat,value,lagged,pred) %>%
  group_by(sm_loc_id) %>%
  summarize(Lon=first(Lon),Lat=first(Lat),
            MSPE=mean((value-pred)^2))

SoilMoisture.all %>% 
ggplot(aes(x=Lon,y=Lat)) +
  geom_point(shape=1) +
  geom_point(data=SoilMoisture.corn.belt.pred.May,
             aes(x=Lon,y=Lat,col=MSPE)) +
  scale_color_viridis_c(direction=1)

### model reliance (MR) (Fisher et al 2018 formula 733) ###

## model reliance code (this takes a long time to run)
## the output from the run is saved as modelreliance.Rdata
## you can just load that and plot

## conducting model reliance for every single pixel changes are too computationally expensive (hence trying MR at cluster level)
SSTclus = read.csv("../SST_withclus.csv", header = T) %>% 
  select(Clust,Lon,Lat)

SST.all.with.clus=SST.all %>%
  left_join(SSTclus)

X=scale(as.matrix(SST.all.with.clus %>% select(starts_with("X"))))
colnames(X)=NULL
row.names(X)=NULL

datelist=unique(SoilMoisture.corn.belt.out$date) ## holdout dates
out.cols=which(SST.lagged.dates %in% datelist) ## columns of the lagged SST data corresponding to hold out dates

### this creates shuffled features for one hold-out timepoint d and SST pixels within one cluster "clus"
### need to run this for all hold out dates and all clusters
perm.MSPE.gen.clust=function(d,clus){
  index=which(SST.all.with.clus$Clust==clus)
  i=which(SST.lagged.dates==d)
  print(c(clus,i))
  ### X.perm is the shuffled feature set using all values for date d for all dates
  X.perm=X
  for(s in index){
    X.perm[s,out.cols]=rep(X[s,i],length(out.cols))
  }
  
  inner.mat.perm = t(X.perm)%*%as.matrix(SST.bas)/nrow(SST.all)
  row.names(inner.mat.perm)=NULL
  SST.inner.perm = cbind(data.frame(date=SST.lagged.dates),as.data.frame(inner.mat.perm))

SoilMoisture.corn.belt.out.perm = SoilMoisture.corn.belt.all %>%
   left_join(SST.inner.perm) %>%
   mutate(month=month(date)) %>%
   dplyr::filter(date > date.in) %>%
   dplyr::filter(!(date==d)) ## the row corresponding to date d is removed from the prediction set (as in formula 3.3 of FIsher et a. 2018 the sum goes over all j \neq i)
  

SoilMoisture.corn.belt.out.pred.perm=SoilMoisture.corn.belt.out.perm %>% 
  group_by(sm_loc_id) %>%
  nest %>% 
  inner_join(SM.corn.belt.output) %>% 
  mutate(pred = map2(fitSM, data, predict)) %>% 
  unnest(c(data,pred)) %>%
  as.data.frame %>%
  select(sm_loc_id,Lon,Lat,date,month,value,lagged,pred)

SoilMoisture.corn.belt.out.pred.perm.May=SoilMoisture.corn.belt.out.pred.perm %>%
  dplyr::filter(month==5)

  c(mean((SoilMoisture.corn.belt.out.pred.perm$pred-SoilMoisture.corn.belt.out.pred.perm$value)^2)/MSPE,
       mean((SoilMoisture.corn.belt.out.pred.perm.May$pred-SoilMoisture.corn.belt.out.pred.perm.May$value)^2)/MSPE.May)
}

### runs perm.MSPE.gen.clust for one cluster for all hold-out dates
MRgen.clust=function(clus){
  MR=as.vector(rowMeans(sapply(datelist,perm.MSPE.gen.clust,clus)))
  SST.MR=SSTclus %>% dplyr::filter(Clust==clus)
  SST.MR$MR=MR[1]
  SST.MR$MR.May=MR[2]
  SST.MR
}

nclust=length(unique(SSTclus$Clust))
l=lapply(1:nclust,MRgen.clust)
MR.df=Reduce('rbind',l)
save(MR.df,file="modelreliance.Rdata")

library(gridExtra)

## plotting the model reliance on a map
## just load the modelreliance.Rdata file to run this
ggplot(data=SST.all,aes(x=Lon,y=Lat)) +
  geom_point() +
geom_point(data=MR.df,aes(x=Lon,y=Lat,col=pmax(0,log(MR.May))),size=1.2) +
  scale_color_gradient2(low = "blue", mid = "white",
                            high = "red", space = "Lab" , midpoint=0.001) +
  geom_hline(yintercept=c(0)) +
  theme_classic() +
  labs(col="")



