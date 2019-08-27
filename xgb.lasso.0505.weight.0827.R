rm(list=ls())
library(xgboost)
library(glmnet)
library(dplyr)
library(Matrix)
library(Metrics) #rmse
library(ggplot2)
library(VIM)
library(corrplot)
library(psych) #skew
library(gridExtra)
library(caret)
library(Ckmeans.1d.dp) #xgboost importance
library(e1071) # svm
train = read.csv("C:/Users/User/Desktop/kaggle/houseprice/train.csv",stringsAsFactors = FALSE)
test = read.csv("C:/Users/User/Desktop/kaggle/houseprice/test.csv",stringsAsFactors = FALSE)
test$SalePrice = NA

tr.te.data = rbind(train,test)

str(tr.te.data)

#################################################################
#################### Data Preprocessing #########################
#################################################################
tr.te.data = tr.te.data %>%
  #some numeric variables transform to factor
  mutate(MSSubClass = as.factor(MSSubClass)) %>% 
  mutate(MoSold = as.factor(MoSold)) %>% 
  mutate(YrSold = as.factor(YrSold)) %>% 
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
  mutate(Alley = as.factor(Alley)) %>% 
  #Transform Score Factor Variables to Numeric Variables
  mutate(ExterQual = replace(ExterQual,ExterQual=="Po",1)) %>% 
  mutate(ExterQual = replace(ExterQual,ExterQual=="Fa",2)) %>% 
  mutate(ExterQual = replace(ExterQual,ExterQual=="TA",3)) %>% 
  mutate(ExterQual = replace(ExterQual,ExterQual=="Gd",4)) %>%
  mutate(ExterQual = replace(ExterQual,ExterQual=="Ex",5)) %>% 
  mutate(ExterQual = as.numeric(ExterQual)) %>% 
  mutate(ExterCond = replace(ExterCond,ExterCond=="Po",1)) %>% 
  mutate(ExterCond = replace(ExterCond,ExterCond=="Fa",2)) %>% 
  mutate(ExterCond = replace(ExterCond,ExterCond=="TA",3)) %>% 
  mutate(ExterCond = replace(ExterCond,ExterCond=="Gd",4)) %>%
  mutate(ExterCond = replace(ExterCond,ExterCond=="Ex",5)) %>% 
  mutate(ExterCond = as.numeric(ExterCond)) %>% 
  mutate(HeatingQC = replace(HeatingQC,HeatingQC=="Po",1)) %>% 
  mutate(HeatingQC = replace(HeatingQC,HeatingQC=="Fa",2)) %>% 
  mutate(HeatingQC = replace(HeatingQC,HeatingQC=="TA",3)) %>% 
  mutate(HeatingQC = replace(HeatingQC,HeatingQC=="Gd",4)) %>%
  mutate(HeatingQC = replace(HeatingQC,HeatingQC=="Ex",5)) %>% 
  mutate(HeatingQC = as.numeric(HeatingQC)) %>% 
  mutate(KitchenQual = replace(KitchenQual,KitchenQual=="Po",1)) %>% 
  mutate(KitchenQual = replace(KitchenQual,KitchenQual=="Fa",2)) %>% 
  mutate(KitchenQual = replace(KitchenQual,KitchenQual=="TA",3)) %>% 
  mutate(KitchenQual= replace(KitchenQual,KitchenQual=="Gd",4)) %>%
  mutate(KitchenQual = replace(KitchenQual,KitchenQual=="Ex",5)) %>% 
  mutate(KitchenQual = as.numeric(KitchenQual)) 

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

#################################################################
#################### Data Visualization #########################
#################################################################

#Highest correlated variable box plot
ggplot(data = com.data[!is.na(com.data$SalePrice),], aes(x = factor(OverallQual), y = SalePrice)) +
  geom_boxplot(col = 'blue') + labs(x = 'Overall Quality') +
  scale_y_continuous(breaks = seq(0, 800000, by = 100000))


#Scatter plot of High correlated variables
ggplot.list = list()

for(i in 1:length(cor.high.names)){
  ggplot.list[[i]] = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes_string(x = cor.high.names[i], y = "SalePrice")) +
    geom_point(col = "slategrey") + 
    geom_smooth(method = "lm", se=FALSE, color="black") +
    scale_y_continuous(breaks= seq(0, 800000, by=100000)) +
    geom_text(hjust = 1, vjust = 2,label = ifelse(com.data[!is.na(com.data$SalePrice),]$GrLivArea>4500,rownames(com.data[!is.na(com.data$SalePrice),],),"") )
}

gg.id = split(1:length(ggplot.list),1:ceiling(length(ggplot.list)/4))

for(i in 1:length(gg.id)){
  grid.arrange(grobs = ggplot.list[gg.id[[i]]], ncol = 2,nrow=2,top = textGrob("High Cor Variables",gp=gpar(fontsize=15,font=1)))
}

#Scatter plot of Low correlated variables
cor.low.names = names(which(sort(cor.numvar[,"SalePrice"], decreasing = TRUE)<=0.5))
ggplot.list = list()

for(i in 1:length(cor.low.names)){
  ggplot.list[[i]] = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes_string(x = cor.low.names[i], y = "SalePrice")) +
    geom_point(col = "slategrey") + 
    geom_smooth(method = "lm", se=FALSE, color="black") +
    scale_y_continuous(breaks= seq(0, 800000, by=100000)) +
    geom_text(hjust = 1, vjust = 2,label = ifelse(com.data[!is.na(com.data$SalePrice),]$GrLivArea>4500,rownames(com.data[!is.na(com.data$SalePrice),],),"") )
  
}

gg.id = split(1:length(ggplot.list),1:ceiling(length(ggplot.list)/4))

for(i in 1:length(gg.id)){
  grid.arrange(grobs = ggplot.list[gg.id[[i]]], ncol = 2,nrow=2,top = textGrob("Low Cor Variables",gp=gpar(fontsize=15,font=1)))
}

#Boxplot of OverallQual Variable
ggplot(data = com.data[!is.na(com.data$SalePrice),], aes(x = as.factor(OverallQual),y=SalePrice))+
  stat_boxplot(geom ='errorbar') + 
  geom_boxplot(fill = 'slategrey',color='darkslategrey')

#Barplot MSSubClass Variable
g1 = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes(x = MSSubClass,y=SalePrice))+
  geom_bar(stat = 'summary', fun.y = 'median', fill = 'slateblue',color='darkslategrey')+
  scale_y_continuous(breaks = seq(0, 800000, by = 50000)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..), size = 3) +
  geom_hline(yintercept = median(com.data$SalePrice,na.rm = TRUE), linetype = 'dashed', color = 'black')

#Countplot of OverallQual Vriable
g2 = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes(x = MSSubClass))+
  geom_bar(stat = 'count',fill = 'slategrey',color='darkslategrey')+
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..), size = 3)

grid.arrange(g1,g2,nrow=2)

#Neighborhood boxplot
g3 = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes(x = Neighborhood,y=SalePrice))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..), size = 3)+
  geom_hline(yintercept=median(com.data$SalePrice[!is.na(com.data$SalePrice)]), linetype="dashed", color = "red")

g4 = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes(x = Neighborhood,y=GrLivArea))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  geom_hline(yintercept=median(com.data$GrLivArea[!is.na(com.data$SalePrice)]), linetype="dashed", color = "red")

grid.arrange(g3,g4,nrow=2)

#Categorical variables boxplot
ggplot.list=list()
cat.var.names = names(which(!sapply(com.data,is.numeric)))
for(i in 1:length(cat.var.names )){
  ggplot.list[[i]] = ggplot(data = com.data[!is.na(com.data$SalePrice),], aes_string(x = cat.var.names[i], y = "SalePrice")) +
    geom_boxplot(col = "slategrey")+
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    geom_label(stat = 'count', aes(label = ..count.., y = ..count..), size = 3)
  
}
gg.id = split(1:length(ggplot.list),1:ceiling(length(ggplot.list)/4))
for(i in 1:length(gg.id)){
  grid.arrange(grobs = ggplot.list[gg.id[[i]]], ncol = 2,nrow=2,top = textGrob("Categorical Variables Boxplot",gp=gpar(fontsize=15,font=1)))
}

#Divide the neighborhood into three districts
N.med.SalPprice = com.data[!is.na(com.data$SalePrice),] %>% group_by(Neighborhood) %>% summarise(median(SalePrice)) %>% as.data.frame()
#poor neighborhood
poor.N.name = N.med.SalPprice[N.med.SalPprice[,2]<quantile(N.med.SalPprice[,2],0.25),1]
for(i in 1:length(poor.N.name)){
  com.data$Neighborhood[com.data$Neighborhood == poor.N.name[i]] = "poor"
}
#middle neighborhood
middle.N.name = N.med.SalPprice[N.med.SalPprice[,2]>=quantile(N.med.SalPprice[,2],0.25) & N.med.SalPprice[,2]<quantile(N.med.SalPprice[,2],0.75),1]
for(i in 1:length(middle.N.name)){
  com.data$Neighborhood[com.data$Neighborhood == middle.N.name[i]] = "middle"
}
#rich neighborhood
rich.N.name = N.med.SalPprice[N.med.SalPprice[,2]>=quantile(N.med.SalPprice[,2],0.75),1]
for(i in 1:length(rich.N.name)){
  com.data$Neighborhood[com.data$Neighborhood == rich.N.name[i]] = "rich"
}
#boxplot
ggplot(data = com.data[!is.na(com.data$SalePrice),], aes(x = Neighborhood,y=SalePrice))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..), size = 3)



#Transform character variables to factor variables
chr.var.names = names(which(sapply(com.data,is.character)))
for(i in 1:length(chr.var.names)){
  com.data[,chr.var.names[i]] = as.factor(com.data[,chr.var.names[i]])
}

#Delete Outlier
com.data = com.data[-c(524,1299),]

#Final numeric data and factoer data
#Remove Multicollinearity Variable
num.var.df = com.data %>% select(cor.high.names,cor.low.names,-GarageCars,-TotRmsAbvGrd,-X1stFlrSF,-GarageYrBlt,-Id)
fac.var.df = com.data %>% select(fac.var.names)
                          


#skewed
skewed.var.name = NULL
for(i in 1:ncol(num.var.df)){
  if(abs(skew(num.var.df[,i]))>0.8){
    num.var.df[,i] = log(1 + num.var.df[,i])
    skewed.var.name = c(skewed.var.name,names(num.var.df)[i])
  }
}

#after log transform, scatterplot
ggplot.list = list()

for(i in 1:length(skewed.var.name)){
  ggplot.list[[i]] = ggplot(data = num.var.df[!is.na(num.var.df$SalePrice),], aes_string(x = skewed.var.name[i], y = "SalePrice")) +
    geom_point(col = "slategrey") + 
    geom_smooth(method = "lm", se=FALSE, color="black") +
    scale_y_continuous(breaks= seq(0, 800000, by=100000)) +
    geom_text(hjust = 1, vjust = 2,label = ifelse(com.data[!is.na(com.data$SalePrice),]$GrLivArea>4500,rownames(com.data[!is.na(com.data$SalePrice),],),"") )
  
}

gg.id = split(1:length(ggplot.list),1:ceiling(length(ggplot.list)/4))

for(i in 1:length(gg.id)){
  grid.arrange(grobs = ggplot.list[gg.id[[i]]], ncol = 2,nrow=2,top = textGrob("Low Cor Variables",gp=gpar(fontsize=15,font=1)))
}

#factor variables one hot encoding(Dummy matrix) 
fac.var.df = as.data.frame(model.matrix(~.,fac.var.df))[,-1]

#check whether some factor are too sparse in the train set
remove.fac.names = names(which(colSums(fac.var.df[is.na(com.data$SalePrice),])<10))
fac.var.df = fac.var.df %>% select(-remove.fac.names)

#complete data
final.com.data = cbind(num.var.df,fac.var.df)
final.train.data = final.com.data[!is.na(final.com.data$SalePrice),]
final.test.data = final.com.data[is.na(final.com.data$SalePrice),]


#################################################################
####################### Model Comparison ########################
#################################################################
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
  #sum of y^2
  exp.y = exp(part.new.valid.data$SalePrice)
  sum.y.2 = sum(exp(part.new.valid.data$SalePrice)^2)
  #lasso
  set.seed(2019)
  lasso.fit = cv.glmnet(as.matrix(part.new.train.data[,-1]),as.matrix(part.new.train.data[,1]))
  lasso.pos = lasso.fit$lambda==lasso.fit$lambda.min
  lasso.beta = rbind(lasso.fit$glmnet.fit$a0,lasso.fit$glmnet.fit$beta)[,lasso.pos]
  lasso.pred = exp(as.matrix(data.frame(1,part.new.valid.data[,-1]))%*%lasso.beta )
  lasso.rmse = sqrt(sum((exp.y - lasso.pred)^2)/sum.y.2 )
  
  #ridge
  set.seed(2019)
  ridge.fit = cv.glmnet(as.matrix(part.new.train.data[,-1]),as.matrix(part.new.train.data[,1]),alpha=0)
  ridge.pos = ridge.fit$lambda==ridge.fit$lambda.min
  ridge.beta = rbind(ridge.fit$glmnet.fit$a0,ridge.fit$glmnet.fit$beta)[,ridge.pos]
  ridge.pred = exp(as.matrix(data.frame(1,part.new.valid.data[,-1]))%*%ridge.beta )
  ridge.rmse = sqrt(sum((exp.y -ridge.pred)^2)/sum.y.2 )
  
  #linear
  set.seed(2019)
  linear.fit = lm(SalePrice~.,part.new.train.data)
  linear.rmse = sqrt(sum((exp.y -exp(predict(linear.fit,part.new.valid.data)))^2)/sum.y.2 )
  
  #xgboost
  part.tr.label = part.new.train.data$SalePrice
  part.vd.label = part.new.valid.data$SalePrice
  pt.d.train = xgb.DMatrix(data=as.matrix(part.new.train.data[,-1]) , label=part.tr.label)
  pt.d.valid = xgb.DMatrix(data=as.matrix(part.new.valid.data[,-1]) , label=part.vd.label)
  
  #xgboost tune
  xgb.grid.search <- expand.grid(max_depth = c(6), #default = 6
                                 eta = 0.01, #default = 0.3
                                 colsample_bytree = c(0.3), #default = 1
                                 subsample = c(1),
                                 alpha = c(1)) #default = 1
  
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
  xgb.rmse = sqrt(sum((exp.y-exp(predict(xgb_mod,pt.d.valid)))^2)/sum.y.2 )
  
  
  eval.vec = c(lasso.rmse,ridge.rmse,linear.rmse,xgb.rmse)
  eval.mat = rbind(eval.mat,eval.vec)
}

#RMSE Boxplot
cm.eval.mat = colMeans(eval.mat) 
colnames(eval.mat) = c("lasso","ridge","linear","xgboost")
stack.eval = stack(data.frame(eval.mat)) ; colnames(stack.eval) = c("RMSE","Methods")

ggplot(data = stack.eval, aes(x=Methods,y=RMSE)) +
  stat_boxplot(geom ='errorbar') + 
  labs(x = 'Each methods') +
  geom_boxplot(fill='slategrey',color='darkslategrey')



#################################################################
##############Fit final model with whole data####################
#################################################################


#linear
linear.fit = lm(SalePrice~.,data=final.train.data)
linear.pred = exp(predict(linear.fit,final.test.data[,-1]))

#ridge
set.seed(2019)
ridge.fit = cv.glmnet(as.matrix(final.train.data[,-1]),as.matrix(final.train.data[,1]),alpha=0)
ridge.beta = rbind(ridge.fit$glmnet.fit$a0,ridge.fit$glmnet.fit$beta)[,ridge.fit$lambda==ridge.fit$lambda.min]
ridge.pred = exp(as.matrix(data.frame(1,final.test.data[,-1]))%*%ridge.beta )
#lasso
set.seed(2019)
lasso.fit = cv.glmnet(as.matrix(final.train.data[,-1]),as.matrix(final.train.data[,1]))
lasso.beta = rbind(lasso.fit$glmnet.fit$a0,lasso.fit$glmnet.fit$beta)[,lasso.fit$lambda==lasso.fit$lambda.min]
lasso.pred = exp(as.matrix(data.frame(1,final.test.data[,-1]))%*%lasso.beta )
#xgboost
tr.label = final.train.data$SalePrice
te.label = final.test.data$SalePrice
d.train = xgb.DMatrix(data=as.matrix(final.train.data[,-1]), label=tr.label)
d.test = xgb.DMatrix(data=as.matrix(final.test.data[,-1]) , label=te.label)

xgb.grid.search <- expand.grid(max_depth = c(6), #default = 6
                               eta = c(0.01), #default = 0.3
                               colsample_bytree = c(0.3), #default = 1
                               subsample = c(1),
                               alpha = c(1)) #default = 1

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

#Xgboost feature importance
mat <- xgb.importance(feature_names = colnames(final.train.data[,-1]),model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:20], rel_to_first = TRUE,top_n=20)



##################################################
##############Final Prediction####################
##################################################

final.submit = 0.5*xgb.pred+0.5*lasso.pred
final.submit = cbind(1461:2919,final.submit)
colnames(final.submit) = c("Id","SalePrice")
write.csv(final.submit,"C:/Users/User/Desktop/kaggle/houseprice/xgb.lasso.0505.weight.0827.csv",row.names = FALSE)
save.image("C:/Users/User/Desktop/kaggle/houseprice/xgb.lasso.0505.weight.0827.RData")
