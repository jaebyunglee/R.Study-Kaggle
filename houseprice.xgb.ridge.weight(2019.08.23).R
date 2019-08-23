rm(list=ls())
library(xgboost)
library(glmnet)
library(dplyr)
library(Matrix)
library(ggplot2)
library(VIM)
library(corrplot)
library(psych) #skew
library(gridExtra)
train = read.csv("C:/Users/User/Desktop/kaggle/houseprice/train.csv",stringsAsFactors = FALSE)
test = read.csv("C:/Users/User/Desktop/kaggle/houseprice/test.csv",stringsAsFactors = FALSE)
test$SalePrice = NA

tr.te.data = rbind(train,test)

str(tr.te.data)

#################################################################
#################### Data Preprocessing #########################
#################################################################
tr.te.data = tr.te.data %>%
              #some numeric variables convert to factor
              mutate(MSSubClass = as.factor(MSSubClass)) %>% 
              mutate(MoSold = as.factor(MoSold)) %>% 
              mutate(YrSold = as.factor(YrSold)) %>% 
              # mutate(OverallQual = as.factor(OverallQual)) %>% 
              # mutate(OverallCond = as.factor(OverallCond)) %>% 
              # mutate(BsmtFullBath = as.factor(BsmtFullBath)) %>% 
              # mutate(BsmtHalfBath = as.factor(BsmtHalfBath)) %>% 
              # mutate(FullBath = as.factor(FullBath)) %>% 
              # mutate(HalfBath = as.factor(HalfBath)) %>% 
              # mutate(KitchenAbvGr = as.factor(KitchenAbvGr)) %>% 
              # mutate(TotRmsAbvGrd = as.factor(TotRmsAbvGrd)) %>% 
              # mutate(Fireplaces = as.factor(Fireplaces)) %>% 
              # mutate(BedroomAbvGr = as.factor(BedroomAbvGr)) %>% 
              # mutate(GarageCars = as.factor(GarageCars)) %>% 
              #some categorical variables Missing Value
              mutate(MiscFeature = replace(MiscFeature,is.na(MiscFeature),"None")) %>% 
              mutate(MiscFeature = as.factor(MiscFeature)) %>% 
              mutate(Fence = replace(Fence,is.na(Fence),"None")) %>% 
              mutate(Fence = as.factor(Fence)) %>% 
              mutate(PoolQC = replace(PoolQC,is.na(PoolQC),"None")) %>% 
              mutate(PoolQC = as.factor(PoolQC)) %>% 
              mutate(GarageCond = replace(GarageCond,is.na(GarageCond),"None")) %>% 
              mutate(GarageCond = as.factor(GarageCond)) %>% 
              mutate(GarageQual = replace(GarageQual,is.na(GarageQual),"None")) %>% 
              mutate(GarageQual = as.factor(GarageQual)) %>% 
              mutate(GarageFinish = replace(GarageFinish,is.na(GarageFinish),"None")) %>% 
              mutate(GarageFinish = as.factor(GarageFinish)) %>% 
              mutate(GarageType = replace(GarageType,is.na(GarageType),"None")) %>% 
              mutate(GarageType = as.factor(GarageType)) %>% 
              mutate(FireplaceQu = replace(FireplaceQu,is.na(FireplaceQu),"None")) %>% 
              mutate(FireplaceQu = as.factor(FireplaceQu)) %>% 
              mutate(BsmtFinType2 = replace(BsmtFinType2,is.na(BsmtFinType2),"None")) %>% 
              mutate(BsmtFinType2 = as.factor(BsmtFinType2)) %>% 
              mutate(BsmtFinType1 = replace(BsmtFinType1,is.na(BsmtFinType1),"None")) %>% 
              mutate(BsmtFinType1 = as.factor(BsmtFinType1)) %>% 
              mutate(BsmtExposure = replace(BsmtExposure,is.na(BsmtExposure),"None")) %>% 
              mutate(BsmtExposure = as.factor(BsmtExposure)) %>% 
              mutate(BsmtCond = replace(BsmtCond,is.na(BsmtCond),"None")) %>% 
              mutate(BsmtCond = as.factor(BsmtCond)) %>% 
              mutate(BsmtQual = replace(BsmtQual,is.na(BsmtQual),"None")) %>% 
              mutate(BsmtQual = as.factor(BsmtQual)) %>% 
              mutate(Alley = replace(Alley,is.na(Alley),"None")) %>% 
              mutate(Alley = as.factor(Alley))


#Variable name with missing values
na.var.names = names(tr.te.data)[colSums(is.na(tr.te.data))!=0]
na.var.names = na.var.names[na.var.names!="SalePrice"]

#MissingValue
com.data = kNN(tr.te.data[,names(tr.te.data)!="SalePrice"], variable = na.var.names, k = 10, imp_var = FALSE )
com.data$SalePrice = tr.te.data$SalePrice

#correration between response and predictors
num.var.names = names(com.data)[sapply(com.data,is.numeric)]
fac.var.names = names(com.data)[!sapply(com.data,is.numeric)]
all.numvar = com.data[,num.var.names]
cor.numvar = cor(all.numvar,use = "pairwise.complete.obs")
cor.high.names = names(which(sort(cor.numvar[,"SalePrice"], decreasing = TRUE)>0.5))
cor.high.mat = cor.numvar[cor.high.names,cor.high.names]
corrplot.mixed(cor.high.mat, tl.col="black",tl.pos = "lt")

#Scatter plot of SalePrice with variables with high correlation
ggplot.list = list()

for(i in 1:length(cor.high.names)){
  ggplot.list[[i]] = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes_string(x = cor.high.names[i], y = "SalePrice")) +
    geom_point(col = "blue") + 
    geom_smooth(method = "lm", se=FALSE, color="black") +
    scale_y_continuous(breaks= seq(0, 800000, by=100000)) +
    geom_text(hjust = 1, vjust = 2,label = ifelse(com.data[!is.na(com.data$SalePrice),]$GrLivArea>4500,rownames(com.data[!is.na(com.data$SalePrice),],),"") )
}

gg.id = split(1:length(ggplot.list),1:ceiling(length(ggplot.list)/4))

for(i in 1:length(gg.id)){
  grid.arrange(grobs = ggplot.list[gg.id[[i]]], ncol = 2,nrow=2,top = textGrob("High Cor Variables",gp=gpar(fontsize=20,font=3)))
}

#Scatter plot of SalePrice with variables with low correlation
cor.low.names = names(which(sort(cor.numvar[,"SalePrice"], decreasing = TRUE)<=0.5))
ggplot.list = list()

for(i in 1:length(cor.low.names)){
  ggplot.list[[i]] = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes_string(x = cor.low.names[i], y = "SalePrice")) +
    geom_point(col = "blue") + 
    geom_smooth(method = "lm", se=FALSE, color="black") +
    scale_y_continuous(breaks= seq(0, 800000, by=100000)) +
    geom_text(hjust = 1, vjust = 2,label = ifelse(com.data[!is.na(com.data$SalePrice),]$GrLivArea>4500,rownames(com.data[!is.na(com.data$SalePrice),],),"") )
  
}

gg.id = split(1:length(ggplot.list),1:ceiling(length(ggplot.list)/4))

for(i in 1:length(gg.id)){
  grid.arrange(grobs = ggplot.list[gg.id[[i]]], ncol = 2,nrow=2,top = textGrob("Low Cor Variables",gp=gpar(fontsize=20,font=3)))
}


#Transform character variables to factor variables
chr.var.names = names(which(sapply(com.data,is.character)))
for(i in 1:length(chr.var.names)){
  com.data[,chr.var.names[i]] = as.factor(com.data[,chr.var.names[i]])
}

#Delete Outlier
com.data = com.data[-c(524,1299),]
#Final num data and remove high correlated predictor 
num.var.df = com.data %>% select(cor.high.names,-X1stFlrSF,-GarageYrBlt)
fac.var.df = com.data %>% select(fac.var.names)

#skewed
for(i in 1:ncol(num.var.df)){
  if(abs(skew(num.var.df[,i]))>0.8){
    num.var.df[,i] = log(1 + num.var.df[,i])
  }
}

#factor variables one hot encoding(Dummy matrix) 
fac.var.df = as.data.frame(model.matrix(~.,fac.var.df))[,-1]

#check if some factor are too sparse in the train set
remove.fac.names = names(which(colSums(fac.var.df[is.na(com.data$SalePrice),])<10))
fac.var.df = fac.var.df %>% select(-remove.fac.names)

#complete data
final.com.data = cbind(num.var.df,fac.var.df)
final.train.data = final.com.data[!is.na(final.com.data$SalePrice),]
final.test.data = final.com.data[is.na(final.com.data$SalePrice),]

##################################################################################################

#train valid split
eval.mat = NULL

for(i in 1:30){
  set.seed(i)
  cat('iteration',i,"/30","\n")
  tr.id = createDataPartition(final.train.data$SalePrice,p=0.7,list=F)
  part.new.train.data = final.train.data[tr.id,]
  part.new.valid.data = final.train.data[-tr.id,]
  
  #eval vec
  eval.vec = NULL
  
  set.seed(2019)
  lasso.fit = cv.glmnet(as.matrix(part.new.train.data[,-1]),as.matrix(part.new.train.data[,1]))
  lasso.rmse = sqrt(sum((part.new.valid.data$SalePrice-exp(predict(lasso.fit,as.matrix(part.new.valid.data[,-1]))))^2))/length(part.new.valid.data$SalePrice)
  
  #ridge
  set.seed(2019)
  ridge.fit = cv.glmnet(as.matrix(part.new.train.data[,-1]),as.matrix(part.new.train.data[,1]),alpha=0)
  ridge.rmse = sqrt(sum((part.new.valid.data$SalePrice-exp(predict(ridge.fit,as.matrix(part.new.valid.data[,-1]))))^2))/length(part.new.valid.data$SalePrice)
  
  #linear
  set.seed(2019)
  linear.fit = lm(SalePrice~.,part.new.train.data)
  linear.rmse = sqrt(sum((part.new.valid.data$SalePrice-exp(predict(linear.fit,part.new.valid.data)))^2))/length(part.new.valid.data$SalePrice)
  
  #xgboost
  part.tr.label = part.new.train.data$SalePrice
  part.vd.label = part.new.valid.data$SalePrice
  pt.d.train = xgb.DMatrix(data=as.matrix(part.new.train.data[,-1]) , label=part.tr.label)
  pt.d.valid = xgb.DMatrix(data=as.matrix(part.new.valid.data[,-1]) , label=part.vd.label)
  
  #xgboost tune
  xgb.grid.search <- expand.grid(max_depth = c(2,6), #default = 6
                                 eta = 0.3, #default = 0.3
                                 colsample_bytree = c(0.7,1), #default = 1
                                 subsample = c(1),
                                 alpha = c(0.2,1)) #default = 1
  
  perf.xgb.mat <- matrix(0,nrow(xgb.grid.search),2)
  colnames(perf.xgb.mat) = c("iter","score")
  
  #grid search
  for(i in 1:nrow(xgb.grid.search)){
    params.xgb<-list(objective = "reg:linear",
                     booster = "gbtree",
                     eta = xgb.grid.search[i,"eta"], #default = 0.3
                     max_depth = xgb.grid.search[i,"max_depth"], #default=6
                     subsample = xgb.grid.search[i,"subsample"],
                     colsample_bytree = xgb.grid.search[i,"colsample_bytree"],
                     alpha = xgb.grid.search[i,"alpha"])
    
    set.seed(2019)
    xgbcv <- xgb.cv(params = params.xgb, data = pt.d.train ,nrounds = 10000,nfold = 5,
                    print_every_n = 50,early_stopping_rounds = 50, maximize = F, verbose = FALSE)
    
    perf.xgb.mat[i,]=c(xgbcv$best_iteration,min(xgbcv$evaluation_log$test_rmse_mean))
  }
  
  
  #find best tuning parameters
  final.perf.xgb.mat = cbind(xgb.grid.search,perf.xgb.mat)
  xgb.opt.par = final.perf.xgb.mat[which.min(final.perf.xgb.mat[,"score"]),]
  
  ###fit xgb
  params.opt.xgb<-list(objective = "reg:linear",
                       booster = "gbtree",
                       eta = xgb.opt.par$eta, #default = 0.3
                       max_depth = xgb.opt.par$max_depth, #default=6
                       subsample = xgb.opt.par$subsample,
                       colsample_bytree = xgb.opt.par$colsample_bytree,
                       alpha = xgb.grid.search[i,"alpha"])
  
  set.seed(2019)
  xgb_mod <- xgb.train(data = pt.d.train , params=params.opt.xgb, nrounds = xgb.opt.par$iter)
  xgb.rmse = sqrt(sum((part.new.valid.data$SalePrice-exp(predict(xgb_mod,pt.d.valid)))^2))/length(part.new.valid.data$SalePrice)
  eval.vec = c(lasso.rmse,ridge.rmse,linear.rmse,xgb.rmse)
  eval.mat = rbind(eval.mat,eval.vec)
}


#################################################################
########################## whole.data ###########################
#################################################################


#ridge
set.seed(2019)
ridge.fit = cv.glmnet(as.matrix(final.train.data[,-1]),as.matrix(final.train.data[,1]),alpha=0)
ridge.pred = exp(predict(ridge.fit, as.matrix(final.test.data[,-1])))
#lasso
set.seed(2019)
lasso.fit = cv.glmnet(as.matrix(final.train.data[,-1]),as.matrix(final.train.data[,1]))
lasso.pred = exp(predict(ridge.fit, as.matrix(final.test.data[,-1])))

#xgboost
tr.label = final.train.data$SalePrice
te.label = final.test.data$SalePrice
d.train = xgb.DMatrix(data=as.matrix(final.train.data[,-1]), label=tr.label)
d.test = xgb.DMatrix(data=as.matrix(final.test.data[,-1]) , label=te.label)

xgb.grid.search <- expand.grid(max_depth = c(2,6), #default = 6
                               eta = 0.3, #default = 0.3
                               colsample_bytree = c(0.7,1), #default = 1
                               subsample = c(1),
                               alpha = c(0.2,1)) #default = 1

perf.xgb.mat <- matrix(0,nrow(xgb.grid.search),2)
colnames(perf.xgb.mat) = c("iter","score")

#grid search
for(i in 1:nrow(xgb.grid.search)){
  cat("xgboost",i,"/",nrow(xgb.grid.search),"\n")
  params.xgb<-list(objective = "reg:linear",
                   booster = "gbtree",
                   eta = xgb.grid.search[i,"eta"], #default = 0.3
                   max_depth = xgb.grid.search[i,"max_depth"], #default=6
                   subsample = xgb.grid.search[i,"subsample"],
                   colsample_bytree = xgb.grid.search[i,"colsample_bytree"],
                   alpha = xgb.grid.search[i,"alpha"])
  
  set.seed(2019)
  xgbcv <- xgb.cv(params = params.xgb, data = d.train ,nrounds = 10000,nfold = 5,
                  print_every_n = 50,early_stopping_rounds = 50, maximize = F, verbose = FALSE)
  
  perf.xgb.mat[i,]=c(xgbcv$best_iteration,min(xgbcv$evaluation_log$test_rmse_mean))
}


#find best tuning parameters
final.perf.xgb.mat = cbind(xgb.grid.search,perf.xgb.mat)
xgb.opt.par = final.perf.xgb.mat[which.min(final.perf.xgb.mat[,"score"]),]

###fit xgb
params.opt.xgb<-list(objective = "reg:linear",
                     booster = "gbtree",
                     eta = xgb.opt.par$eta, #default = 0.3
                     max_depth = xgb.opt.par$max_depth, #default=6
                     subsample = xgb.opt.par$subsample,
                     colsample_bytree = xgb.opt.par$colsample_bytree,
                     alpha = xgb.grid.search[i,"alpha"])

set.seed(2019)
xgb_mod <- xgb.train(data = d.train , params=params.opt.xgb, nrounds = xgb.opt.par$iter)
xgb.pred = exp(predict(xgb_mod,d.test))

final.submit = 0.7*ridge.pred+0.3*xgb.pred
final.submit = cbind(1461:2919,final.submit)
colnames(final.submit) = c("Id","SalePrice")
write.csv(final.submit,"C:/Users/User/Desktop/kaggle/houseprice/xgb.ridge.weight.csv",row.names = FALSE)



