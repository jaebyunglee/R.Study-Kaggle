# -----------------------------------------------------------------------------------------
# 파일명 : House Price Prediction.R
# 설  명 : 집값 예측 (Lasso, Random forest, SVM)
# 일  자 : 2020/03/13
# 작성자 : 이재병 분석가(jblee@begas.co.kr, 010-2775-0930)
# 패키지 : ggplt2, randomForest, caret, xgboost, e1071, corplot, gridExtra, glmnet, Matrixm caret
# -----------------------------------------------------------------------------------------

#작업공간
setwd("C:/Rproject")

#library 불러오기
library(ggplot2) #ggplot함수
library(randomForest) #randomforest함수
library(caret) #createfolds 함수
library(xgboost)
library(e1071) #svm 함수
library(Hmisc) #describe 함수
library(corrplot) #corrplot 함수
library(gridExtra) #grid.arrange 함수
library(glmnet) #cv.glmnet 함수
library(Matrix) #model.matrix 함수


#데이터 불러오기
train_df <- read.csv("DAT/train.csv", header = T, sep = ",", stringsAsFactors = F)
test_df <- read.csv("DAT/test.csv", header = T, sep = ",", stringsAsFactors = F)
#Id 변수 제거
train_df$Id <- NULL; test_df$Id <- NULL; test_df$SalePrice <- NA
all_df <- rbind(train_df,test_df)
#데이터 살펴보기
head(train_df);head(test_df)
str(train_df);str(test_df)
Hmisc::describe(train_df)

# -----------------------------------------------------------------------------------------
# 데이터 전처리
# -----------------------------------------------------------------------------------------

### Missing value 탐색 ###

# 전체 데이터 Missing value
na_num <- colSums(is.na(all_df))/nrow(all_df)*100
na_num[na_num!=0]
# Train 데이터 Missing value
tr_na_num <- colSums(is.na(train_df))/nrow(train_df)*100
tr_na_num[tr_na_num!=0]
# Test 데이터 Missing value
te_na_num <- colSums(is.na(test_df))/nrow(test_df)*100
te_na_num[te_na_num!=0]
# Train 데이터에만 있는 Missing Value -> Electrical 변수(0.06%)  <<<제거>>>
base::setdiff(names(tr_na_num[tr_na_num!=0]), names(te_na_num[te_na_num!=0]))
# Test 데이터에만 있는 Missing Value
base::setdiff(names(te_na_num[te_na_num!=0]), names(tr_na_num[tr_na_num!=0]))
rm(na_num,tr_na_num,te_na_num)

#범주형 변수 명
car.var.names <- names(which(sapply(all_df,is.character)))
car.var.names <- c(car.var.names,"MSSubClass","OverallQual","OverallCond","BsmtFullBath","BsmtHalfBath","GarageCars")
not.car.var.names <- setdiff(names(all_df),car.var.names)
#숫자형 변수 명
num.var.names <- not.car.var.names[not.car.var.names!="SalePrice"]


### 범주형 Missing value imputation ###


#Test에만 결측이 있는경우 -> 최빈값 대체
#결측치 수가 30개 미만 -> 최빈값 대체
#결측치수가 30개 이상 -> "N"이라는 새로운 범주로 대체
#결측치가 "None"이라는 범주의 의미 -> "None" 범주로 대체

all_df$MSZoning[is.na(all_df$MSZoning)] <- "RL" #결측값 4개, <<<테스트에만 NA 존재, 최빈값으로 대체>>>
all_df$Alley[is.na(all_df$Alley)] <- "N" #골목 유형, NA가 골목이 없음을 의미, <<<범주 "N"으로 대체>>>
all_df$Utilities[is.na(all_df$Utilities)] <- "AllPub" #결측값 2개 <<<테스트에만 NA 존재, 최빈값으로 대체>>>,범주 구분이 없어 제거해야할듯
all_df$Exterior1st[is.na(all_df$Exterior1st)] <- "VinylSd" #집 외장재 종류, 결측값 1개 <<<테스트에만 NA 존재, 최빈값으로 대체>>>
all_df$Exterior2nd[is.na(all_df$Exterior2nd)] <- "VinylSd" #집 외장재 종류, 결측값 1개 <<<테스트에만 NA 존재, 최빈값으로 대체>>>
all_df$MasVnrType[is.na(all_df$MasVnrType)] <- "None" #석재 종류, 결측값 24개, 최빈값으로 대체
all_df$BsmtQual[is.na(all_df$BsmtQual)] <- "None" #결측값 81개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$BsmtCond[is.na(all_df$BsmtCond)] <- "None" #결측값 82개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$BsmtExposure[is.na(all_df$BsmtExposure)] <- "None" #결측값 82개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$BsmtFinType1[is.na(all_df$BsmtFinType1)] <- "None" #결측값 79개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$BsmtFinType2[is.na(all_df$BsmtFinType2)] <- "None" #결측값 80개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$Electrical[is.na(all_df$Electrical)] <- "SBrkr" #결측값 1개, <<<최빈값으로 대체>>>
all_df$BsmtFullBath[is.na(all_df$BsmtFullBath)] <- 0 #결측값 2개, <<<최빈값으로 대체>>> 
all_df$BsmtHalfBath[is.na(all_df$BsmtHalfBath)] <- 0 #결측값 2개, <<<최빈값으로 대체>>>
all_df$KitchenQual[is.na(all_df$KitchenQual)] <- "Gd" #결측값 1개, <<<테스트에만 NA 존재, 최빈값으로 대체>>>
all_df$Functional[is.na(all_df$Functional)] <- "Typ" #결측값 2개, <<<테스트에만 NA 존재, 최빈값으로 대체>>>
all_df$FireplaceQu[is.na(all_df$FireplaceQu)] <- "N" #<<<NA가 None 범주를 의미, 범주 "N"으로 대체>>>
all_df$GarageType[is.na(all_df$GarageType)] <- "N" #결측값 157개 , <<<N 범주로 대체>>>
all_df$GarageFinish[is.na(all_df$GarageFinish)] <- "N" #결측값 159개 , <<<N 범주로 대체>>>
all_df$GarageCars[is.na(all_df$GarageCars)] <- 2 #결측값 1개, <<<테스트에만 NA 존재, 최빈값으로 대체>>>
all_df$GarageQual[is.na(all_df$GarageQual)] <- "None" #결측값 159개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$GarageCond[is.na(all_df$GarageCond)] <- "None" #결측값 159개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$PoolQC[is.na(all_df$PoolQC)] <- "N" #결측값 2909개, 수영장이 있는가 없는가를 의미 <<<범주 "None"으로 대체>>>
all_df$Fence[is.na(all_df$Fence)] <- "None" #결측값 2348개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$MiscFeature[is.na(all_df$MiscFeature)] <- "None" #결측값 2814개, <<<NA가 None 범주를 의미, 범주 "None"으로 대체>>>
all_df$SaleType[is.na(all_df$SaleType)] <- "WD" #결측값 1개, <<<테스트에만 NA 존재, 최빈값으로 대체>>>

#범주형 변수 Factor로 변환
for(i in car.var.names){
  all_df[,i] <- as.factor(all_df[,i])
}
rm(i)


### 숫자형 Missing value imputation ###


#결측값 30개 미만 Median으로 대체
all_df$MasVnrArea[is.na(all_df$MasVnrArea)] <- median(all_df$MasVnrArea,na.rm = T)
all_df$BsmtFinSF1[is.na(all_df$BsmtFinSF1)] <- median(all_df$BsmtFinSF1,na.rm = T)
all_df$BsmtFinSF2[is.na(all_df$BsmtFinSF2)] <- median(all_df$BsmtFinSF2,na.rm = T)
all_df$BsmtUnfSF[is.na(all_df$BsmtUnfSF)] <- median(all_df$BsmtUnfSF,na.rm = T)
all_df$TotalBsmtSF[is.na(all_df$TotalBsmtSF)] <- median(all_df$TotalBsmtSF,na.rm = T)
all_df$GarageArea[is.na(all_df$GarageArea)] <- median(all_df$GarageArea,na.rm = T)

#결측값 30개 이상인 경우 -> 상관계수가 높은 10개 변수를 독립변수로하여 회귀분석 대체(LotFrontage:486개, GarageYrBlt:159개)
cor_mat <- cor(all_df[,num.var.names],use="pairwise.complete.obs",method = "pearson") #숫자형 변수 상관계수 행렬

lot.order <- order(abs(cor_mat[,"LotFrontage"]),decreasing = T) #LotFrontage와 상관계수가 높은변수 상위 10개
lot.cor.high.var.names <- names(cor_mat[lot.order,"LotFrontage"])[1:11]
lot.lm <- lm(LotFrontage ~ ., data = all_df[!is.na(all_df$LotFrontage),lot.cor.high.var.names]) #LotFrontage를 종속변수로 회귀적합
all_df$LotFrontage[is.na(all_df$LotFrontage)] <- predict(lot.lm,all_df[is.na(all_df$LotFrontage),lot.cor.high.var.names]) #예측

gar.order <- order(abs(cor_mat[,"GarageYrBlt"]),decreasing = T) #GarageYrBlt와 상관계수가 높은변수 상위 10개
gar.cor.high.var.names <- names(cor_mat[gar.order,"GarageYrBlt"])[1:11]
gar.lm <- lm(GarageYrBlt ~ ., data = all_df[!is.na(all_df$GarageYrBlt),gar.cor.high.var.names]) #GarageYrBlt를 족송변수로 회귀적합
all_df$GarageYrBlt[is.na(all_df$GarageYrBlt)] <- predict(gar.lm,all_df[is.na(all_df$GarageYrBlt),gar.cor.high.var.names]) #예측
rm(cor_mat, gar.lm, lot.lm, gar.cor.high.var.names, gar.order, lot.cor.high.var.names, lot.order,not.car.var.names)


# -----------------------------------------------------------------------------------------
# 탐색적 분석
# -----------------------------------------------------------------------------------------


### 숫자형 독립변수 EDA


#숫자형 독립변수들 간의 상관계수
cor_mat <- cor(all_df[!is.na(all_df$SalePrice),num.var.names])
corrplot(cor_mat, order = "hclust", addrect = 4, rect.col = "red")
rm(cor_mat)

#종속변수와 관련이 높은 독립변수 상관계수
cor_mat <- cor(all_df[!is.na(all_df$SalePrice),c("SalePrice",num.var.names)])
cor.high.ind <- order(abs(cor_mat[,"SalePrice"]),decreasing = T)
corrplot(cor_mat[cor.high.ind,cor.high.ind])
cor_mat[cor.high.ind,"SalePrice"]
rm(cor_mat,cor.high.ind)

#종속변수 histogram, boxplot
png("OUT/EDA_Y_SalesPrice.png", width = 1024, height = 768)
p1 <- ggplot(all_df[!is.na(all_df$SalePrice),],aes_string(x="SalePrice")) + geom_histogram(colour="black",bins=30)
p1 <- p1 + labs(y = "Count") + ggtitle("Histogram of SalePrice") + theme(plot.title = element_text(hjust = 0.5))
p2 <- ggplot(all_df[!is.na(all_df$SalePrice),],aes_string(x=factor(" "),y="SalePrice")) + geom_boxplot(colour = "black",fill="darkgray")
p2 <- p2 + ggtitle("Boxplot of SalePrice") + theme(plot.title = element_text(hjust = 0.5))
grid.arrange(p1,p2,ncol=2)
dev.off()
rm(p1,p2)

#숫자형 독립변수 histogram, boxplot, Scatterplot
for(g.i in num.var.names){
  png(paste("OUT/EDA_NUM_",g.i,".png",sep=""), width = 1024, height = 768)
  p1 <- ggplot(all_df[!is.na(all_df$SalePrice),],aes_string(x=g.i)) + geom_histogram(colour="black",bins=30)
  p1 <- p1 + labs(y = "Count") + ggtitle(paste("Histogram of ", g.i,sep = "")) + theme(plot.title = element_text(hjust = 0.5))
  p2 <- ggplot(all_df[!is.na(all_df$SalePrice),],aes_string(x=factor(" "),y=g.i)) + geom_boxplot(colour = "black",fill="darkgray")
  p2 <- p2 + ggtitle(paste("Boxplot of ", g.i,sep = "")) + theme(plot.title = element_text(hjust = 0.5))
  p3 <- ggplot(all_df[!is.na(all_df$SalePrice),], aes_string(x = g.i, y = "SalePrice")) + geom_point(col='blue') 
  p3 <- p3 + geom_smooth(method = "lm", color="black") + ggtitle(paste("Scatterplot of ",g.i,sep="")) + theme(plot.title = element_text(hjust = 0.5))
  grid.arrange(arrangeGrob(p3), arrangeGrob(p1,p2, ncol=1), ncol=2, widths=c(2, 1))
  dev.off()
}
rm(g.i,p1,p2,p3)


### 범주형 독립변수 EDA


for(g.i in car.var.names){
  png(paste("OUT/EDA_CAT_",g.i,".png",sep=""), width = 1024, height = 768)
  p1 <- ggplot(all_df[!is.na(all_df$SalePrice),],aes_string(x = g.i, y = "SalePrice")) + geom_boxplot()
  p1 <- p1 + geom_hline(yintercept=median(all_df$SalePrice,na.rm = T), linetype="dashed", color = "red")
  p1 <- p1 + ggtitle(paste("Boxplot of SalePrice ~ ",g.i,sep = "")) + theme(plot.title = element_text(hjust = 0.5))
  grid.arrange(p1)
  dev.off()
}
rm(g.i,p1)


### 주요 변수 추출


# 숫자형 독립변수 (Lasso 로 추출)
set.seed(2019)
lasso.mat <- as.matrix(all_df[!is.na(all_df$SalePrice),c("SalePrice",num.var.names)])
cv.glmnet <- glmnet::cv.glmnet(lasso.mat[,-1],lasso.mat[,1])
lasso.beta <- cv.glmnet$glmnet.fit$beta[,cv.glmnet$lambda==cv.glmnet$lambda.min]
imp.num.var.names <- names(which(lasso.beta != 0))
rm(lasso.mat, lasso.beta, cv.glmnet)

# 범주형 독립변수 (Random Forest 로 추출)

set.seed(2019)
formula.x <- paste("SalePrice~",paste(car.var.names,collapse = "+"),sep = "")
rf_output <- randomForest::randomForest(as.formula(formula.x),data=all_df[!is.na(all_df$SalePrice),])
rf_imp <- randomForest::importance(rf_output)
rf_imp_df <- data.frame(var = rownames(rf_imp), imp = rf_imp, stringsAsFactors = F)
rf_imp_df <- rf_imp_df[order(rf_imp_df$IncNodePurity,decreasing = T),]
imp.car.var.names <- rf_imp_df[1:15,"var"]
rm(formula.x, rf_imp, rf_imp_df, rf_output)

imp.var.names <- c(imp.num.var.names, imp.car.var.names) #최종 X변수
rm(imp.num.var.names, imp.car.var.names, car.var.names, num.var.names)


# -----------------------------------------------------------------------------------------
# 예측 분석 : Lasso, Random forest, SVM
# -----------------------------------------------------------------------------------------

# 최종 예측용 데이터
f.train_x <- all_df[!is.na(all_df$SalePrice),imp.var.names]
f.test_x <- all_df[is.na(all_df$SalePrice),imp.var.names]
f.train_y <- all_df$SalePrice[!is.na(all_df$SalePrice)]
f.test_y <- all_df$SalePrice[is.na(all_df$SalePrice)]
formula.x <- paste("SalePrice~",paste(imp.var.names,collapse = "+"),sep = "") 

### Lasso ###
set.seed(2019)
lasso <- cv.glmnet(model.matrix(~.,data=f.train_x)[,-1], f.train_y)
lasso.pred <- predict(lasso, model.matrix(~.,data=f.test_x)[,-1], s = lasso$lambda.min)
lasso.pred2 <- predict(lasso, model.matrix(~.,data=f.train_x)[,-1], s = lasso$lambda.min)

#제출 
lasso.submit <- data.frame(Id = rownames(lasso.pred), SalePrice = as.vector(lasso.pred))
write.csv(lasso.submit, "OUT/lasso.submit.csv", row.names = F)

#실제값과 예측값 비교
plot(f.train_y, lasso.pred2)
rm(train_df, test_df, imp.var.names, lasso.pred2, lasso.submit)


### randomforest ###
f.train <- all_df[!is.na(all_df$SalePrice),c(imp.var.names,"SalePrice")]
f.test <- all_df[is.na(all_df$SalePrice),c(imp.var.names,"SalePrice")]

set.seed(2019)
fold.id <- createFolds(y = f.train_y, k = 5, list = T, returnTrain = F)
rf.grid <- expand.grid(mtry = c(2,8,16))
rf.perf.mat <- matrix(0, nrow = nrow(rf.grid))
colnames(rf.perf.mat) <- "score"
for(g in 1:nrow(rf.grid)){
  e.vec <- NULL
  for(f in 1:5){
    rf <- randomForest::randomForest(as.formula(formula.x),data=f.train[-fold.id[[f]],], mtry = rf.grid[g,"mtry"])
    e.vec <- c(e.vec,sum((predict(rf, f.train[fold.id[[f]],]) - f.train[fold.id[[f]],"SalePrice"])^2)/length(f.train[fold.id[[f]],"SalePrice"]))
  }
  rf.perf.mat[g,] <- mean(e.vec)
}
final.perf.mat <- cbind(rf.grid, rf.perf.mat)
opt.mtry <- final.perf.mat[which.min(final.perf.mat$score),"mtry"]

rf <- randomForest::randomForest(as.formula(formula.x), data=f.train , mtry = opt.mtry)
rf.pred <- predict(rf, f.test)
rf.pred2 <- predict(rf, f.train)

#제출 
rf.submit <- data.frame(Id = names(rf.pred), SalePrice = as.vector(rf.pred))
write.csv(rf.submit, "OUT/rf.submit.csv", row.names = F)

#실제값과 예측값 비교
plot(f.train$SalePrice, rf.pred2)
rm(rf.grid, rf.perf.mat, g, f, final.perf.mat, opt.mtry, e.vec, rf.pred2, rf.submit)


### svm ###
set.seed(2019)
svm.grid <- expand.grid(cost = 4^(-3:3), gamma = 4^(-3:3))
svm.perf.mat = matrix(0,nrow(svm.grid),1)
colnames(svm.perf.mat) = "score"

for(g in 1:nrow(svm.perf.mat)){
  cat("svm tune :",g,"/",nrow(svm.perf.mat),"\n")
  err.vec = NULL
  for(f in 1:5){
    svm <- e1071::svm(as.formula(formula.x),data=f.train[-fold.id[[f]],], type = "eps-regression", cost = svm.grid[g,"cost"], gamma = svm.grid[g,"gamma"])
    err.vec <- c(err.vec,sum((predict(svm, f.train[fold.id[[f]],]) - f.train[fold.id[[f]],"SalePrice"])^2)/length(f.train[fold.id[[f]],"SalePrice"]))
  }
  svm.perf.mat[g,] <- mean(err.vec)
}

final.perf.mat <- cbind(svm.grid, svm.perf.mat)
opt.par <- final.perf.mat[which.min(final.perf.mat$score),c("cost","gamma")]
svm <- e1071::svm(as.formula(formula.x),data=f.train, type="eps-regression", cost = opt.par$cost, gamma = opt.par$gamma)
svm.pred <- predict(svm, subset(f.test, select = -SalePrice))
svm.pred2 <- predict(svm, f.train)

#제출 
svm.submit <- data.frame(Id = names(svm.pred), SalePrice = as.vector(svm.pred))
write.csv(svm.submit, "OUT/svm.submit.csv", row.names = F)

#실제값과 예측값 비교
plot(f.train$SalePrice, svm.pred2)
rm(g, gamma, f, err.vec, f.test_y, f.train_y, formula.x)



### ensemble ###
ensemble.submit <- data.frame(Id = names(svm.pred), SalePrice = as.vector((svm.pred+rf.pred+lasso.pred)/3))
write.csv(svm.submit, "OUT/ensemble.submit.csv", row.names = F)
