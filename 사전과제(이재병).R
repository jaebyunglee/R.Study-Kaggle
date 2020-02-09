library(ggplot2);library(dplyr);library(ggpubr);library(text2vec)
library(RColorBrewer);library(VIM);library(caret);library(Matrix)
library(glmnet);library(randomForest);library(xgboost);library(e1071)
library(keras); library(stringr)
rm(list=ls())

set.seed(2019)
#########################
#####데이터 불러오기#####
#########################
work.dir = "C:\\Users\\jaebyung.lee\\Desktop\\"
train.data = read.table(paste(work.dir,"train.csv",sep = ""),header = T,sep = ",")
test.data = read.table(paste(work.dir,"test.csv",sep = ""),header = T,sep = ",")
test.data$Survived = NA
full.data = rbind(train.data,test.data)
full.data$Survived = as.factor(full.data$Survived)
full.data$Pclass = as.factor(full.data$Pclass)
full.data$Name = as.character(full.data$Name)

#########################
#######데이터 탐색#######
#########################
#각 변수별 NA의 개수
colSums(apply(full.data,2,is.na))

###Survived 변수
#생존 342명 사망 549명 으로 생존보다 사망이 약 1.6배 많다.
ggplot(full.data[!is.na(full.data$Survived),], aes(x = Survived, fill = Survived)) +
  geom_bar(stat='count', position='dodge') + theme_bw() +
  labs(x = '생존과 사망의 빈도') + geom_label(stat='count', aes(label=..count..))+
  geom_text(aes(label=scales::percent(..count../sum(..count..))),
            stat='count',position=position_fill(vjust=0.7),vjust=-1,hjust=0.4)

###Sex
#탑승자의 성별
#남자 577명 여자 314명
#여성의 경우 74%생존 한 것에 비해 남성의 경우 약 19%만 생존
sg1=ggplot(full.data[!is.na(full.data$Survived),], aes(x = Sex, fill = Sex)) +
  geom_bar(stat='count', position='dodge') + theme_bw() +
  labs(x = '성별 빈도') + geom_label(stat='count', aes(label=..count..))+
  geom_text(aes(label=scales::percent(..count../sum(..count..))),
            stat='count',position=position_fill(vjust=0.7),vjust=-1,hjust=0.4)

sg2=ggplot(full.data[!is.na(full.data$Survived),], aes(x = Sex, fill = Survived)) +
  geom_bar(stat='count', position='dodge') + theme_bw() + labs(x = '성별 생존 사망 수') +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))

sg3=ggplot(full.data[!is.na(full.data$Survived),], aes(x = Sex, fill = Survived)) +
  labs(x = '성별 생존 사망 비율', y = "Percent") + geom_bar( position='fill') + theme_bw() +
  geom_text(aes(label=scales::percent(..count../tapply(..count.., ..x.. ,sum)[..x..])),
            stat='count',position=position_fill(vjust=0.5))
ggarrange(sg1,ggarrange(sg2,sg3,ncol = 2),nrow = 2)

###Pclass
#티켓의 등급
#1:일등석(216명), 2:이등석(184명), 3:삼등석(491명)
#1,2,3 등급 순으로 생존율이 낮아지며, 3등급의 경우 전체 인원 대비 사망자의 수가 약 42%
#티켓의 등급별, 남여별 비율을 살펴보면 여성의 경우 1,2등급 대다수가 생존, 남성의 경우 2,3등급 대다수가 사망
pg1=ggplot(full.data[!is.na(full.data$Survived),], aes(x = Pclass, fill = Pclass)) +
  geom_bar(stat='count', position='dodge') + theme_bw() +
  labs(x = '티켓 등급별 빈도') + geom_label(stat='count', aes(label=..count..))+
  geom_text(aes(label=scales::percent(..count../sum(..count..))),
            stat='count',position=position_fill(vjust=0.7),vjust=-1,hjust=0.4)

pg2=ggplot(full.data[!is.na(full.data$Survived),], aes(x = Pclass, fill = Survived)) +
  geom_bar(stat='count', position='dodge') + theme_bw() +
  labs(x = '티켓 등급별 생존 사망 수') +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))

pg3=ggplot(full.data[!is.na(full.data$Survived),], aes(x = Pclass, fill = Survived)) +
  labs(x = '티켓 등급별 생존 사망 비율', y = "Percent") +
  geom_bar( position='fill') + theme_bw() +
  geom_text(aes(label=scales::percent(..count../tapply(..count.., ..x.. ,sum)[..x..])),
            stat='count',position=position_fill(vjust=0.5))

pg4=ggplot(full.data[!is.na(full.data$Survived),], aes(x = Pclass, fill = Survived)) +
  labs(x = '성별 및 티켓 등급별 생존 사망 비율', y = "Percent") +
  geom_bar( position='fill') + theme_bw() +
  facet_grid(.~Sex)

ggarrange(pg1,pg2,pg3,pg4,nrow = 2,ncol=2)


###Age
#263개의 결측값 존재, 중앙값, knn 또는 회귀 등의 예측값으로 결측치 처리 예정
#5세 단위로 구분하여 생존율 확인

sum(is.na(full.data$Age))
full.data$AgeFactor = NA
full.data$AgeFactor[full.data$Age<=5] = "0-5"
full.data$AgeFactor[full.data$Age>5 & full.data$Age<=10 & !is.na(full.data$Age)] = "5-10"
full.data$AgeFactor[full.data$Age>10 & full.data$Age<=15& !is.na(full.data$Age)] = "10-15"
full.data$AgeFactor[full.data$Age>15 & full.data$Age<=20& !is.na(full.data$Age)] = "15-20"
full.data$AgeFactor[full.data$Age>20 & full.data$Age<=25& !is.na(full.data$Age)] = "20-25"
full.data$AgeFactor[full.data$Age>25 & full.data$Age<=30& !is.na(full.data$Age)] = "25-30"
full.data$AgeFactor[full.data$Age>30 & full.data$Age<=35& !is.na(full.data$Age)] = "30-35"
full.data$AgeFactor[full.data$Age>35 & full.data$Age<=40& !is.na(full.data$Age)] = "35-40"
full.data$AgeFactor[full.data$Age>40 & full.data$Age<=45& !is.na(full.data$Age)] = "40-45"
full.data$AgeFactor[full.data$Age>45 & full.data$Age<=50& !is.na(full.data$Age)] = "45-50"
full.data$AgeFactor[full.data$Age>50 & full.data$Age<=55& !is.na(full.data$Age)] = "50-55"
full.data$AgeFactor[full.data$Age>55 & full.data$Age<=60& !is.na(full.data$Age)] = "55-60"
full.data$AgeFactor[full.data$Age>60& !is.na(full.data$Age)] = "60>"
full.data$AgeFactor = factor(full.data$AgeFactor,levels = c("0-5","5-10","10-15","15-20","20-25","25-30","30-35","35-40","40-45","45-50","50-55","55-60","60>"))

ag1 = ggplot(full.data[!is.na(full.data$Survived)&!is.na(full.data$Age),],aes(x=Age))+ 
  geom_histogram(fill="#96CDCD",binwidth = 5,colour = "black") + theme_bw() + labs(x = "나이 분포")


ag2=ggplot(full.data[!is.na(full.data$Survived)&!is.na(full.data$AgeFactor),], aes(x = AgeFactor, fill = Survived)) +
  labs(x = '연령별 생존 사망 비율', y = "Percent") +
  geom_bar( position='fill') + theme_bw() + 
  geom_text(aes(label=scales::percent(..count../tapply(..count.., ..x.. ,sum)[..x..])),
            stat='count',position=position_fill(vjust=0.5)) +theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

ag3=ggplot(full.data[!is.na(full.data$Survived)&!is.na(full.data$AgeFactor),], aes(x = AgeFactor, fill = Survived)) +
  labs(x = '성별 및 연령별 생존 사망 비율', y = "Percent") +
  geom_bar( position='fill') + theme_bw() +  facet_grid(.~Sex) +theme(axis.text.x = element_text(angle = 45, hjust = 1)) 


ggarrange(ggarrange(ag1,ag2),ag3,nrow=2)

###SibSp & Parch
#형제자매, 부모자식을 모두 합쳐서 가족 수 라는 변수를 만듬
full.data$Family = full.data$SibSp + full.data$Parch

fg1=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(SibSp), fill = Survived)) +
  labs(x = '함께 탑승한 형제, 배우자 수별 생존 사망 수') +
  geom_bar(stat='count', position='dodge') + theme_bw() +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))

fg2=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(SibSp), fill = Survived)) +
  labs(x = '함께 탑승한 형제, 배우자 수별 생존 사망 비율', y = "Percent") +
  geom_bar( position='fill') + theme_bw() +
  geom_text(aes(label=scales::percent(..count../tapply(..count.., ..x.. ,sum)[..x..])),
            stat='count',position=position_fill(vjust=0.5))

fg3=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(SibSp), fill = Survived)) +
  labs(x = '좌석 등급에 따른 함께 탑승한 형제, 배우자 수별 생존 사망 수') +
  geom_bar(stat='count', position='dodge') + theme_bw() + facet_grid(.~Pclass) +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))

fg4=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(Parch), fill = Survived)) +
  labs(x = '함께 탑승한 자녀, 부모 수별 생존 사망 수') +
  geom_bar(stat='count', position='dodge') + theme_bw() +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))

fg5=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(Parch), fill = Survived)) +
  labs(x = '함께 탑승한 자녀, 부모 수별 생존 사망 비율', y = "Percent") +
  geom_bar( position='fill') + theme_bw() +
  geom_text(aes(label=scales::percent(..count../tapply(..count.., ..x.. ,sum)[..x..])),
            stat='count',position=position_fill(vjust=0.5))

fg6=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(Parch), fill = Survived)) +
  labs(x = '좌석 등급에 따른 함께 탑승한 자녀, 부모 수별 생존 사망 수') +
  geom_bar(stat='count', position='dodge') + theme_bw() + facet_grid(.~Pclass) +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))


fg7=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(Family), fill = Survived)) +
  labs(x = '함께 탑승한 가족 수별 생존 사망 수') +
  geom_bar(stat='count', position='dodge') + theme_bw() +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))

fg8=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(Family), fill = Survived)) +
  labs(x = '함께 탑승한 가족 수별 생존 사망 비율', y = "Percent") +
  geom_bar( position='fill') + theme_bw() +
  geom_text(aes(label=scales::percent(..count../tapply(..count.., ..x.. ,sum)[..x..])),
            stat='count',position=position_fill(vjust=0.5))

fg9=ggplot(full.data[!is.na(full.data$Survived),], aes(x = as.factor(Family), fill = Survived)) +
  labs(x = '좌석 등급에 따른 함께 탑승한 가족 수별 생존 사망 수') +
  geom_bar(stat='count', position='dodge') + theme_bw() + facet_grid(.~Pclass) +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))


ggarrange(fg1,fg2,fg3,fg4,fg5,fg6,nrow=2,ncol=3)
ggarrange(fg7,fg8,fg9,ncol=3)
###Fare
sum(is.na(full.data$Fare))
full.data$Fare.a = full.data$Fare/(full.data$SibSp+full.data$Parch)
#요금이 낮은경우 생존과 사망의 분포가 확연한 차이를 보임
#다만 요금이 낮으면 선실 등급이 낮을것이기 때문에 선실등급에 따른 분포를 확인
ffg1=ggplot(full.data[!is.na(full.data$Survived),], aes(x = Fare, fill = Survived)) + 
  geom_density(alpha=0.5, aes(fill=factor(Survived)),colour="black") + theme_bw()

#1등급, 2등급, 3등급 선실로 갈수록 요금이 낮은경우 사망한 경우가 많을것이라 판단
ffg2=ggplot(full.data[!is.na(full.data$Survived) & full.data$Pclass == 1,], aes(x = Fare, fill = Survived)) + 
  geom_histogram(alpha=0.5, aes(fill=factor(Survived)),position = "fill",colour="black") + theme_bw() + labs(y = "Percent")
ffg3=ggplot(full.data[!is.na(full.data$Survived) & full.data$Pclass == 2,], aes(x = Fare, fill = Survived)) + 
  geom_histogram(alpha=0.5, aes(fill=factor(Survived)),position = "fill",colour="black") + theme_bw() + labs(y = "Percent")
ffg4=ggplot(full.data[!is.na(full.data$Survived) & full.data$Pclass == 3,], aes(x = Fare, fill = Survived)) + 
  geom_histogram(alpha=0.5, aes(fill=factor(Survived)),position = "fill",colour="black") + theme_bw() + labs(y = "Percent")



ggarrange(ffg1,ffg2,ffg3,ffg4,nrow = 2,ncol=2,
          labels = c("Pclass-all","Pclass-1","Pclass-2","Pclass-3"),
          font.label = list(size = 9,color="cyan4"),hjust = 0,vjust=1)


###Embarked
#Embarked가 없는 데이터 2개 존재
#탑승한 곳에 따라서 생존과 사망의 비율이 다소 차이를 보임
#C , Q , S 순으로 생존률이 낮아짐
eg1=ggplot(full.data[!is.na(full.data$Survived)&full.data$Embarked!="",],aes(x=Embarked, fill=Survived))+
  geom_bar(position="fill") + labs(y = "Percent",x = "탑승 장소에 따른 생존 비율") +
  geom_text(aes(label=scales::percent(..count../tapply(..count.., ..x.. ,sum)[..x..])),
            stat='count',position=position_fill(vjust=0.5))

eg2=ggplot(full.data[!is.na(full.data$Survived)&full.data$Embarked!="",],aes(x=Pclass,fill=Sex))+
  geom_bar(stat = "count",position="dodge")  + facet_grid(.~Embarked) + labs(x = "탑승 장소, 성별, 선실등급 빈도") +
  geom_text(aes(label=..count..),stat='count',position=position_dodge(0.9))

ggarrange(eg1,eg2,ncol = 2)

###Cabin, Ticket, Passengerid
sum(full.data$Cabin=="")
full.data$Ticket
full.data$PassengerId

#########################
######데이터 전처리######
#########################

###Embarked 결측값 처리(2개)
#Fare 결측값 처리(1개) median으로 채워넣음 
full.data$Fare[is.na(full.data$Fare)] = median(full.data$Fare,na.rm=T)

###Embarked 결측값 처리(2개) mode로 채워 넣음
full.data$Embarked[full.data$Embarked==""] = as.factor("S")
full.data$Embarked = factor(full.data$Embarked, levels = c("C","Q","S"))

###Age 결측값 처리(263개)
#Pclass, Sex, SibSp, parch, Fare, Emarked 변수로 Age를 예측 
var.name = c("Pclass","Sex","SibSp","Parch","Fare","Age","Embarked")
Age.train=full.data[!is.na(full.data$Age),var.name]
Age.test=full.data[is.na(full.data$Age),var.name]

###median
full.data$med.Age = full.data$Age
full.data[is.na(full.data$med.Age),"med.Age"] = median(full.data$Age,na.rm =T) 
###knn
#5-fold cv를 사용하여 opt k를 찾음
fold.id = createFolds(y=Age.train$Age,k=5)

k.vec = c(1:30)
result.mat = matrix(NA,length(k.vec),2)
colnames(result.mat) = c("k","score")
for(j in 1:length(k.vec)){
  err.vec = NULL
  for(i in 1:5){
    temp.Age.train = Age.train; temp.Age.train[fold.id[[i]],"Age"] = NA
    imp.Age=kNN(temp.Age.train, variable = "Age" , k = k.vec[j])
    err.vec=c(err.vec,mean((imp.Age[fold.id[[i]],"Age"] - Age.train[fold.id[[i]],"Age"])^2))
  }
  result.mat[j,1] = k.vec[j]; result.mat[j,2] = mean(err.vec)
}

opt.k = result.mat[which.min(result.mat[,2]),"k"]
knn.Age = kNN(full.data[,var.name], variable = "Age" , k=opt.k)

#결측값이 모두 대체된 Age변수를 데이터에 삽입
full.data$knn.Age = knn.Age$Age

###regression
full.data$lm.Age = full.data$Age
lm.fit=lm(Age~.,data=Age.train)
full.data[is.na(full.data$lm.Age),"lm.Age"] = predict(lm.fit,Age.test)
full.data$lm.Age[full.data$lm.Age<=0] = min(full.data$Age,na.rm = T)

###random forest
full.data$rf.Age = full.data$Age
rf.grid.search = expand.grid(mtry = c(1,2,3,4,5))
perf.rf.mat <- matrix(0,nrow(rf.grid.search),1)
colnames(perf.rf.mat) = c("score")
#cross - valid
for(j in 1:nrow(rf.grid.search)){
  rf.cv.err = NULL
  for(i in 1:5){
    temp.Age.train = Age.train; temp.Age.train[fold.id[[i]],"Age"] = NA
    rf = randomForest(Age~.,data=temp.Age.train[-fold.id[[i]],],
                      mtry = rf.grid.search[i,"mtry"],ntree=500)
    rfcr = mean((Age.train[fold.id[[i]],"Age"]-predict(rf,temp.Age.train[fold.id[[i]],-6]))^2)
    rf.cv.err = rbind(rf.cv.err,rfcr)
  }
  perf.rf.mat[j,] = colMeans(rf.cv.err)
}

final.perf.mat=cbind(rf.grid.search,perf.rf.mat)
rf.opt.par=final.perf.mat[which.min(final.perf.mat$score),"mtry"]
rf = randomForest(Age~.,data=Age.train, mtry = rf.grid.search[i,"mtry"],ntree=500)
full.data[is.na(full.data$rf.Age),"rf.Age"] = predict(rf,Age.test)


#가족인 경우 확인
sort(table(full.data$Ticket),decreasing = T)
full.data %>% filter(Ticket=="19950")
full.data %>% filter(Ticket=="347088")
full.data %>% filter(Ticket=="367226")

###Family Group 변수 추가
#이름에서 성을 추출
name=str_split(full.data$Name,pattern = ",",n=2,simplify = T)[,1]
full.data$Name = name

#성과 티켓 고유 번호를 합침
family.group=paste(full.data$Name, as.character(full.data$Ticket),sep="")
temp.G.F=model.matrix(~family.group)

#학습데이터에서 0인 경우, 테스트 데이터에서 0인경우 제외
Family.G=temp.G.F[,(colSums(temp.G.F[!is.na(full.data$Survived),])!=0) * (colSums(temp.G.F[is.na(full.data$Survived),])!=0)==1]

#Family size
full.data$FamilySize = NA
full.data$FamilySize[full.data$Family==0] = "A"
full.data$FamilySize[full.data$Family>=1 & full.data$Family<=3] = "B"
full.data$FamilySize[full.data$Family>=4] = "C"
full.data$FamilySize = as.factor(full.data$FamilySize)

#######################
######데이터 분석######
#######################
#Age 데이터셋 4개
sp1=full.data[!is.na(full.data$Survived),c("Pclass","Sex","med.Age","SibSp","Parch","Fare","Embarked")]
sp2=full.data[!is.na(full.data$Survived),c("Pclass","Sex","lm.Age","SibSp","Parch","Fare","Embarked")]
sp3=full.data[!is.na(full.data$Survived),c("Pclass","Sex","knn.Age","SibSp","Parch","Fare","Embarked")]
sp4=full.data[!is.na(full.data$Survived),c("Pclass","Sex","rf.Age","SibSp","Parch","Fare","Embarked")]

df1 = model.matrix(~.,sp1)[,-1];df2 = model.matrix(~.,sp2)[,-1]
df3 = model.matrix(~.,sp3)[,-1];df4= model.matrix(~.,sp4)[,-1]

ran.num = 20; num.methods = 5 ; folds = 5
result.mat = matrix(0,4,num.methods)
colnames(result.mat) = c("lasso","RF","XGB","SVM","NN")
rownames(result.mat) = c("med.Age","lm.Age","knn.Age","rf.Age")
##############randomization##################
for(ran in 1:ran.num){
  cat("#############",ran,"##################\n")
  train.id = createDataPartition(y=train.data$Survived,p=0.7)
  cvIndex = createFolds(train.data$Survived[train.id$Resample1], folds, returnTrain = F)
  temp.mat = NULL
  for(u in 1:4){
    cat("@@@@@@",u,"@@@@@@\n")
    if(u==1){temp.data=df1}else if(u==2){temp.data=df2}else if(u==3){temp.data=df3}else{temp.data=df4}
    result.vec = NULL
    train.x = temp.data[train.id$Resample1,]
    train.y = train.data$Survived[train.id$Resample1]
    test.x = temp.data[-train.id$Resample1,]
    test.y = train.data$Survived[-train.id$Resample1]
    
    #####################################Logistic lasso##################################
    foldid = rep(NA,length(train.y))
    for(m in 1:folds){ foldid[cvIndex[[m]]] = m}
    lasso.fit = cv.glmnet(train.x,train.y,family="binomial",foldid = foldid)
    lasso.pred = (predict(lasso.fit,test.x,s=lasso.fit$lambda.min)>0)+0
    result.vec = c(result.vec,mean(test.y==(lasso.pred)))
    cat("lasso finish",ran,"\n")
    #####################################Random forest###################################
    rf.grid.search = expand.grid(mtry = c(3,5,8))
    perf.rf.mat <- matrix(0,nrow(rf.grid.search),1)
    colnames(perf.rf.mat) = c("score")
    #cross - valid
    for(i in 1:nrow(rf.grid.search)){
      rf.cv.err = NULL
      for(ind in 1:folds){
        rf = randomForest(as.matrix(train.x[-cvIndex[[ind]],]),as.factor(train.y[-cvIndex[[ind]]]),
                          mtry = rf.grid.search[i,"mtry"],ntree=500)
        rfcr = mean(train.y[cvIndex[[ind]]]!=predict(rf,train.x[cvIndex[[ind]],]))
        rf.cv.err = rbind(rf.cv.err,rfcr)
      }
      perf.rf.mat[i,] = colMeans(rf.cv.err)
    }
    
    #find best tuning parameters
    final.perf.rf.mat = cbind(rf.grid.search,perf.rf.mat)
    rf.opt.par = final.perf.rf.mat[which.min(final.perf.rf.mat[,"score"]),"mtry"]
    #random Forest final fit
    rf.fit = randomForest(as.matrix(train.x),as.factor(train.y),mtry = rf.opt.par,ntree=500,importance = T)
    result.vec=c(result.vec,mean(test.y==predict(rf.fit,test.x)))
    cat("rf finish",ran,"\n")
    ##########################################xgboost#########################################
    d.train = xgb.DMatrix(data=train.x,label=train.y)
    d.test = xgb.DMatrix(data=test.x,label=test.y)
    ### xgb tune
    #xgboost grid
    xgb.grid = expand.grid(eta = c(0.01,0.001),lambda = c(0.1,0.05),
                           max_depth = c(6,12,18),min_child_weight = c(0.1,0.05),
                           gamma = c(0.3),subsample = c(0.7,1),colsample_bytree = c(0.7,1))
    xgb.perf.mat = matrix(0,nrow(xgb.grid),2)
    colnames(xgb.perf.mat) = c("iter","score")
    #grid search
    for(i in 1:nrow(xgb.grid)){
      params.xgb<-list(objective = "binary:logistic",booster = "gbtree",
                       gamma = xgb.grid[i,"gamma"],eta = xgb.grid[i,"eta"], 
                       max_depth = xgb.grid[i,"max_depth"],min_child_weight = xgb.grid[i,"min_child_weight"], 
                       subsample = xgb.grid[i,"subsample"],colsample_bytree = xgb.grid[i,"colsample_bytree"],
                       lambda = xgb.grid[i,"lambda"])
      xgbcv <- xgb.cv(params = params.xgb, data = d.train,nrounds = 1000,folds = cvIndex,
                      print_every_n = 50,early_stopping_rounds = 50, maximize = F, verbose = FALSE)
      xgb.perf.mat[i,]=c(xgbcv$best_iteration,min(xgbcv$evaluation_log$test_error_mean))
    }
    
    #find best tuning parameters
    final.perf.xgb.mat = cbind(xgb.grid,xgb.perf.mat)
    xgb.opt.par = final.perf.xgb.mat[which.min(final.perf.xgb.mat[,"score"]),]
    
    params.xgb<-list(objective = "binary:logistic",booster = "gbtree",
                     gamma = xgb.opt.par$gamma,eta = xgb.opt.par$eta, 
                     max_depth = xgb.opt.par$max_depth,min_child_weight = xgb.opt.par$min_child_weight,
                     subsample = xgb.opt.par$subsample,colsample_bytree = xgb.opt.par$colsample_bytree,
                     lambda = xgb.opt.par$lambda)
    #final xgboost with optimal parameters
    xgboost = xgb.train(data = d.train, params=params.xgb, nrounds = xgb.opt.par$iter)
    result.vec=c(result.vec,mean(test.y==(predict(xgboost,d.test)>0.5)+0))
    cat("xgbfinish",ran,"\n")
    ##########################################SVM#########################################
    svm.grid = expand.grid(gamma = 2^(-10:10), cost = 2^(-10:10))
    svm.perf.mat = matrix(0,nrow(svm.grid),1)
    colnames(svm.perf.mat) = c("score")
    for(i in 1:nrow(svm.grid)){
      svm.cv.err = NULL
      for(ind in 1:folds){
        svm = svm(as.matrix(train.x[-cvIndex[[ind]],]),as.factor(train.y[-cvIndex[[ind]]]),
                  gamma = svm.grid[i,"gamma"],cost = svm.grid[i,"cost"])
        svmcr=mean(train.y[cvIndex[[ind]]]!=predict(svm,as.matrix(train.x[cvIndex[[ind]],])))
        svm.cv.err = rbind(svm.cv.err,svmcr)
      }
      svm.perf.mat[i,] = colMeans(svm.cv.err)
    }
    #find best tuning parameters
    final.perf.svm.mat = cbind(svm.grid,svm.perf.mat)
    svm.opt.par = final.perf.svm.mat[which.min(final.perf.svm.mat[,"score"]),]
    #final svm with optimal parameters
    svm = svm(as.matrix(train.x),as.factor(train.y),gamma=svm.opt.par$gamma,cost=svm.opt.par$cost)
    result.vec = c(result.vec,mean(test.y==predict(svm,as.matrix(test.x))))
    cat("svm finish",ran,"\n")
    ##########################################NN#########################################
    num_epochs = 100
    # 5 fold cross validation
    cv.err.mat = NULL
    for(i in 1:folds){
      val_data = as.matrix(train.x[cvIndex[[i]],]) ; partial_train_data = as.matrix(train.x[-cvIndex[[i]],])
      val_target = train.y[cvIndex[[i]]] ; partial_train_target = train.y[-cvIndex[[i]]]
      #build model
      model = keras_model_sequential() %>% 
        layer_dense(units = 16, activation = "relu", input_shape = dim(train.x)[2]) %>% #shape is demension
        layer_dense(unit = 8,activation = "relu") %>% 
        layer_dense(units = 1, activation = "sigmoid")
      #compile model
      model %>% compile(optimizer = optimizer_adam(lr=0.001),loss = "binary_crossentropy",metrics = c("accuracy"))
      #save each epochs accuracy
      history = model %>% fit(partial_train_data,partial_train_target,
                              validation_data = list(val_data,val_target), #val data
                              epochs = num_epochs, batchsize = 1, verbose = 0)
      cv.err.mat = rbind(cv.err.mat,history$metrics$val_acc)
    }
    opt.epochs = which.max(colMeans(cv.err.mat))
    
    
    ###build model with optmal parameters
    model = keras_model_sequential() %>% 
      layer_dense(units = 16, activation = "relu", input_shape = dim(train.x)[2]) %>% 
      layer_dense(unit = 8,activation = "relu") %>% 
      layer_dense(units = 1, activation = "sigmoid")
    ###compile model
    model %>% compile(optimizer = optimizer_adam(lr=0.001),loss = "binary_crossentropy",metrics = c("accuracy")
    )
    ### final NN model with optimal tuning parameters
    model %>% fit(as.matrix(train.x),train.y, #val data
                  epochs = opt.epochs, batchsize = 1, verbose = 0)
    result.vec = c(result.vec,mean(test.y==(predict(model,as.matrix(test.x))>0.5)+0))
    cat("nn finish",ran,"\n")
    ###result.mat
    temp.mat = rbind(temp.mat,result.vec)
  }
  result.mat = result.mat + temp.mat
}

result.mat/ran.num

save.image(paste(work.dir,"titanic_random_result.mat.RData",sep = ""))


#####################
######최종 예측######
#####################
folds = 5
sp=full.data[,c("Pclass","Sex","rf.Age","SibSp","Parch","Fare","Embarked","FamilySize")]
final.data = model.matrix(~.,sp)[,-1]
# #Family.G 와 결합
# final.data = cbind(final.data,Family.G)
final.train.x = final.data[!is.na(full.data$Survived),]
final.train.y = train.data$Survived
final.test.x = final.data[is.na(full.data$Survived),]
final.test.y = rep(0,dim(final.test.x)[1])
f.cvIndex = createFolds(final.train.y, folds, returnTrain = F)


#######################Random Forest###########################
rf.grid.search = expand.grid(mtry = c(3,5,8))
perf.rf.mat <- matrix(0,nrow(rf.grid.search),1)
colnames(perf.rf.mat) = c("score")

#cross - valid
for(i in 1:nrow(rf.grid.search)){
  rf.cv.err = NULL
  for(ind in 1:folds){
    rf = randomForest(as.matrix(final.train.x[-f.cvIndex[[ind]],]),as.factor(final.train.y[-f.cvIndex[[ind]]]),
                      mtry = rf.grid.search[i,"mtry"],ntree=500)
    rfcr = mean(final.train.y[f.cvIndex[[ind]]]!=predict(rf,final.train.x[f.cvIndex[[ind]],]))
    rf.cv.err = rbind(rf.cv.err,rfcr)
  }
  perf.rf.mat[i,] = colMeans(rf.cv.err)
}

#find best tuning parameters
final.perf.rf.mat = cbind(rf.grid.search,perf.rf.mat)
rf.opt.par = final.perf.rf.mat[which.min(final.perf.rf.mat[,"score"]),"mtry"]
#random Forest final fit
rf.fit = randomForest(as.matrix(final.train.x),as.factor(final.train.y),mtry = rf.opt.par,ntree=500)
#final prediction
final.pred=predict(rf.fit,as.matrix(final.test.x))
final.submit=cbind(as.numeric(rownames(final.test.x)),as.numeric(final.pred)-1)
colnames(final.submit) = c("PassengerId","Survived")
#submit
write.csv(final.submit,paste(work.dir,"RF.Fsize.submit.csv"),row.names = FALSE)


####################### Xgboost ###########################

d.train = xgb.DMatrix(data=final.train.x,label=final.train.y)
d.test = xgb.DMatrix(data=final.test.x,label=final.test.y)
### xgb tune
#xgboost grid
xgb.grid = expand.grid(eta = c(0.01,0.001),lambda = c(0.1,0.05),
                       max_depth = c(6,12,18),min_child_weight = c(0.1,0.05),
                       gamma = c(0.3,0.5),subsample = c(0.7,1),colsample_bytree = c(0.7,1))
xgb.perf.mat = matrix(0,nrow(xgb.grid),2)
colnames(xgb.perf.mat) = c("iter","score")
#grid search
for(i in 1:nrow(xgb.grid)){
  print(i)
  params.xgb<-list(objective = "binary:logistic",booster = "gbtree",
                   gamma = xgb.grid[i,"gamma"],eta = xgb.grid[i,"eta"], #default = 0.3
                   max_depth = xgb.grid[i,"max_depth"],min_child_weight = xgb.grid[i,"min_child_weight"], 
                   subsample = xgb.grid[i,"subsample"],colsample_bytree = xgb.grid[i,"colsample_bytree"],
                   lambda = xgb.grid[i,"lambda"])
  xgbcv <- xgb.cv(params = params.xgb, data = d.train,nrounds = 1000,folds = f.cvIndex,
                  print_every_n = 50,early_stopping_rounds = 50, maximize = F, verbose = FALSE)
  xgb.perf.mat[i,]=c(xgbcv$best_iteration,min(xgbcv$evaluation_log$test_error_mean))
}

#find best tuning parameters
final.perf.xgb.mat = cbind(xgb.grid,xgb.perf.mat)
xgb.opt.par = final.perf.xgb.mat[which.min(final.perf.xgb.mat[,"score"]),]

params.xgb<-list(objective = "binary:logistic",booster = "gbtree",
                 gamma = xgb.opt.par$gamma,eta = xgb.opt.par$eta, #default = 0.3
                 max_depth = xgb.opt.par$max_depth,min_child_weight = xgb.opt.par$min_child_weight,
                 subsample = xgb.opt.par$subsample,colsample_bytree = xgb.opt.par$colsample_bytree,
                 lambda = xgb.opt.par$lambda)

#final xgboost with optimal parameters
xgboost = xgb.train(data = d.train, params=params.xgb, nrounds = xgb.opt.par$iter)


#final prediction
final.pred=(predict(xgboost,d.test)>0.5)+0
final.submit=cbind(as.numeric(rownames(final.test.x)),final.pred)
colnames(final.submit) = c("PassengerId","Survived")

#submit
write.csv(final.submit,paste(work.dir,"xgb.Fsize.submit.csv"),row.names = FALSE)


####################### SVM ###########################

svm.grid = expand.grid(gamma = 2^(-10:10), cost = 2^(-10:10))
svm.perf.mat = matrix(0,nrow(svm.grid),1)
colnames(svm.perf.mat) = c("score")
for(i in 1:nrow(svm.grid)){
  svm.cv.err = NULL
  for(ind in 1:folds){
    svm = svm(as.matrix(final.train.x[-f.cvIndex[[ind]],]),as.factor(final.train.y[-f.cvIndex[[ind]]]),
              gamma = svm.grid[i,"gamma"],cost = svm.grid[i,"cost"])
    svmcr=mean(final.train.y[f.cvIndex[[ind]]]!=predict(svm,as.matrix(final.train.x[f.cvIndex[[ind]],])))
    svm.cv.err = rbind(svm.cv.err,svmcr)
  }
  svm.perf.mat[i,] = colMeans(svm.cv.err)
}
#find best tuning parameters
final.perf.svm.mat = cbind(svm.grid,svm.perf.mat)
svm.opt.par = final.perf.svm.mat[which.min(final.perf.svm.mat[,"score"]),]
#final svm with optimal parameters
svm.model = svm(as.matrix(final.train.x),as.factor(final.train.y),gamma=svm.opt.par$gamma,cost=svm.opt.par$cost,probability=T)


#final prediction
final.pred=as.numeric(predict(svm.model,as.matrix(final.test.x)))-1
final.submit=cbind(as.numeric(rownames(final.test.x)),final.pred)
colnames(final.submit) = c("PassengerId","Survived")

#submit
write.csv(final.submit,paste(work.dir,"svm.Fsize.submit.csv"),row.names = FALSE)


#######################lasso###########################


foldid = rep(NA,length(final.train.y))
for(m in 1:folds){ foldid[f.cvIndex[[m]]] = m}
lasso.fit = cv.glmnet(final.train.x,final.train.y,family="binomial",foldid = foldid)
lasso.pred = (predict(lasso.fit,final.test.x,s=lasso.fit$lambda.min)>0)+0

#final prediction
final.submit=cbind(as.numeric(rownames(final.test.x)),lasso.pred)
colnames(final.submit) = c("PassengerId","Survived")

#submit
write.csv(final.submit,paste(work.dir,"lasso.Fsize.submit.csv"),row.names = FALSE)

#######################Neural Network###########################


num_epochs = 100
# 5 fold cross validation
cv.err.mat = NULL
for(i in 1:folds){
  cat("nn fold",i,"\n")
  val_data = as.matrix(final.train.x[f.cvIndex[[i]],]) ; partial_train_data = as.matrix(final.train.x[-f.cvIndex[[i]],])
  val_target = final.train.y[f.cvIndex[[i]]] ; partial_train_target = final.train.y[-f.cvIndex[[i]]]
  #build model
  model = keras_model_sequential() %>% 
    layer_dense(units = 64,activation = "relu", input_shape = dim(final.train.x)[2]) %>% #demension
    layer_dense(unit = 16,activation = "relu") %>% 
    layer_dense(unit = 8,activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")
  #compile model
  model %>% compile(optimizer = optimizer_adam(lr=0.001),loss = "binary_crossentropy",metrics = c("accuracy"))
  #save each epochs accuracy
  history = model %>% fit(partial_train_data,partial_train_target,
                          validation_data = list(val_data,val_target), #val data
                          epochs = num_epochs, batchsize = 1, verbose = 0)
  cv.err.mat = rbind(cv.err.mat,history$metrics$val_acc)
}
opt.epochs = which.max(colMeans(cv.err.mat))

###build model with optmal parameters
model = keras_model_sequential() %>% 
  layer_dense(units = 64,activation = "relu", input_shape = dim(final.train.x)[2]) %>% #demension
  layer_dense(unit = 16,activation = "relu") %>% 
  layer_dense(unit = 8,activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
###compile model
model %>% compile(optimizer = optimizer_adam(lr=0.001),loss = "binary_crossentropy",metrics = c("accuracy")
)

### final NN model with optimal tuning parameters
model %>% fit(as.matrix(final.train.x),final.train.y, #val data
              epochs = opt.epochs, batchsize = 1, verbose = 0)

nn.pred = (predict(model,as.matrix(final.test.x))>0.5)+0


#final prediction
final.submit=cbind(as.numeric(rownames(final.test.x)),nn.pred)
colnames(final.submit) = c("PassengerId","Survived")

#submit
write.csv(final.submit,paste(work.dir,"nn.Fsize.submit.csv"),row.names = FALSE)


####################### Ensemble ###########################
rf=as.numeric(predict(rf.fit,as.matrix(final.test.x)))-1
xgb=(predict(xgboost,d.test)>0.5)+0
svm=as.numeric(predict(svm.model,as.matrix(final.test.x)))-1
lasso=(predict(lasso.fit,final.test.x,s=lasso.fit$lambda.min)>0)+0
nn=(predict(model,as.matrix(final.test.x))>0.5)+0

ens.pred=(apply(cbind(rf,xgb,nn,lasso,svm),1,sum)>=3)+0
final.submit=cbind(as.numeric(rownames(final.test.x)),ens.pred)
colnames(final.submit) = c("PassengerId","Survived")
#submit
write.csv(final.submit,paste(work.dir,"ensemble.Fsize.submit.csv"),row.names = FALSE)


save.image(paste(work.dir,"titanic_random1.RData",sep = ""))




####################################
#####남 여 모델 따로 최종 예측######
####################################
#Fare, Cabin 변환
folds = 5
sp=full.data[,c("Pclass","Sex","rf.Age","SibSp","Parch","Fare","Embarked","FamilySize")]
final.data = model.matrix(~.,sp)[,-1]
# final.data = cbind(final.data,Family.G)
final.train.x = final.data[!is.na(full.data$Survived),]
final.train.y = train.data$Survived
final.test.x = final.data[is.na(full.data$Survived),]
final.test.y = rep(0,dim(final.test.x)[1])
f.cvIndex = createFolds(final.train.y, folds, returnTrain = F)

#male
male.final.train.x = final.train.x[final.train.x[,"Sexmale"]==1,-3]
male.final.train.y = train.data$Survived[final.train.x[,"Sexmale"]==1]
male.final.test.x =  final.test.x[final.test.x[,"Sexmale"]==1,-3]
male.final.test.y = rep(0,dim(male.final.test.x )[1])
male.f.cvIndex = createFolds(male.final.train.y, folds, returnTrain = F)

#female
female.final.train.x = final.train.x[final.train.x[,"Sexmale"]==0,-3]
female.final.train.y = train.data$Survived[final.train.x[,"Sexmale"]==0]
female.final.test.x =  final.test.x[final.test.x[,"Sexmale"]==0,-3]
female.final.test.y = rep(0,dim(female.final.test.x )[1])
female.f.cvIndex = createFolds(female.final.train.y, folds, returnTrain = F)


################################ RF ######################################
rf.grid.search = expand.grid(mtry = c(3,5,8))
male.perf.rf.mat <- matrix(0,nrow(rf.grid.search),1)
female.perf.rf.mat <- matrix(0,nrow(rf.grid.search),1)
colnames(male.perf.rf.mat) = c("score")
colnames(female.perf.rf.mat) = c("score")
#cross - valid
for(i in 1:nrow(rf.grid.search)){
  male.rf.cv.err = NULL
  female.rf.cv.err = NULL
  for(ind in 1:folds){
    male.rf = randomForest(as.matrix(male.final.train.x[-male.f.cvIndex[[ind]],]),as.factor(male.final.train.y[-male.f.cvIndex[[ind]]]),
                           mtry = rf.grid.search[i,"mtry"],ntree=500)
    male.rfcr = mean(male.final.train.y[male.f.cvIndex[[ind]]]!=predict(male.rf,male.final.train.x[male.f.cvIndex[[ind]],]))
    male.rf.cv.err = rbind(male.rf.cv.err,male.rfcr)
    
    female.rf = randomForest(as.matrix(female.final.train.x[-female.f.cvIndex[[ind]],]),as.factor(female.final.train.y[-female.f.cvIndex[[ind]]]),
                             mtry = rf.grid.search[i,"mtry"],ntree=500)
    female.rfcr = mean(female.final.train.y[female.f.cvIndex[[ind]]]!=predict(female.rf,female.final.train.x[female.f.cvIndex[[ind]],]))
    female.rf.cv.err = rbind(female.rf.cv.err,female.rfcr)
  }
  male.perf.rf.mat[i,] = colMeans(male.rf.cv.err)
  female.perf.rf.mat[i,] = colMeans(female.rf.cv.err)
}

#find best tuning parameters
male.final.perf.rf.mat = cbind(rf.grid.search,male.perf.rf.mat)
male.rf.opt.par = male.final.perf.rf.mat[which.min(male.final.perf.rf.mat[,"score"]),"mtry"]
female.final.perf.rf.mat = cbind(rf.grid.search,female.perf.rf.mat)
female.rf.opt.par = female.final.perf.rf.mat[which.min(female.final.perf.rf.mat[,"score"]),"mtry"]
#random Forest final fit
male.rf.fit = randomForest(as.matrix(male.final.train.x),as.factor(male.final.train.y),mtry = male.rf.opt.par,ntree=500)
female.rf.fit = randomForest(as.matrix(female.final.train.x),as.factor(female.final.train.y),mtry = female.rf.opt.par,ntree=500)
#final prediction
male.final.pred=predict(male.rf.fit,as.matrix(male.final.test.x))
female.final.pred=predict(female.rf.fit,as.matrix(female.final.test.x))

#submit
sex.RF.pred=as.numeric(c(male.final.pred,female.final.pred)[rownames(final.test.x)])-1
final.submit=cbind(as.numeric(rownames(final.test.x)),sex.RF.pred)
colnames(final.submit) = c("PassengerId","Survived")
write.csv(final.submit,paste(work.dir,"RF.Sex.Fsize.submit.csv"),row.names = FALSE)


################################ SVM ######################################

svm.grid = expand.grid(gamma = 2^(-10:10), cost = 2^(-10:10))
male.svm.perf.mat = matrix(0,nrow(svm.grid),1)
female.svm.perf.mat = matrix(0,nrow(svm.grid),1)
colnames(male.svm.perf.mat) = c("score")
colnames(female.svm.perf.mat) = c("score")
for(i in 1:nrow(svm.grid)){
  male.svm.cv.err = NULL
  female.svm.cv.err = NULL
  for(ind in 1:folds){
    #male
    male.svm = svm(as.matrix(male.final.train.x[-male.f.cvIndex[[ind]],]),as.factor(male.final.train.y[-male.f.cvIndex[[ind]]]),
                   gamma = svm.grid[i,"gamma"],cost = svm.grid[i,"cost"])
    male.svmcr=mean(male.final.train.y[male.f.cvIndex[[ind]]]!=predict(male.svm,as.matrix(male.final.train.x[male.f.cvIndex[[ind]],])))
    male.svm.cv.err = rbind(male.svm.cv.err,male.svmcr)
    #female
    female.svm = svm(as.matrix(female.final.train.x[-female.f.cvIndex[[ind]],]),as.factor(female.final.train.y[-female.f.cvIndex[[ind]]]),
                     gamma = svm.grid[i,"gamma"],cost = svm.grid[i,"cost"])
    female.svmcr=mean(female.final.train.y[female.f.cvIndex[[ind]]]!=predict(female.svm,as.matrix(female.final.train.x[female.f.cvIndex[[ind]],])))
    female.svm.cv.err = rbind(female.svm.cv.err,female.svmcr)
  }
  male.svm.perf.mat[i,] = colMeans(male.svm.cv.err)
  female.svm.perf.mat[i,] = colMeans(female.svm.cv.err)
}
#find best tuning parameters
male.final.perf.svm.mat = cbind(svm.grid,male.svm.perf.mat)
female.final.perf.svm.mat = cbind(svm.grid,female.svm.perf.mat)
male.svm.opt.par = male.final.perf.svm.mat[which.min(male.final.perf.svm.mat[,"score"]),]
female.svm.opt.par = female.final.perf.svm.mat[which.min(female.final.perf.svm.mat[,"score"]),]
#final svm with optimal parameters
male.svm.model = svm(as.matrix(male.final.train.x),as.factor(male.final.train.y),gamma=male.svm.opt.par$gamma,cost=male.svm.opt.par$cost,probability=T)
female.svm.model = svm(as.matrix(female.final.train.x),as.factor(female.final.train.y),gamma=female.svm.opt.par$gamma,cost=female.svm.opt.par$cost,probability=T)
#final prediction
male.final.pred=predict(male.svm.model,as.matrix(male.final.test.x))
female.final.pred=predict(female.svm.model,as.matrix(female.final.test.x))

#submit
sex.svm.pred=as.numeric(c(male.final.pred,female.final.pred)[rownames(final.test.x)])-1
final.submit=cbind(as.numeric(rownames(final.test.x)),sex.svm.pred)
colnames(final.submit) = c("PassengerId","Survived")
write.csv(final.submit,paste(work.dir,"svm.Sex.Fsize.submit.csv"),row.names = FALSE)

######################### Xgboost ##############################

male.d.train = xgb.DMatrix(data=male.final.train.x,label=male.final.train.y)
male.d.test = xgb.DMatrix(data=male.final.test.x,label=male.final.test.y)
female.d.train = xgb.DMatrix(data=female.final.train.x,label=female.final.train.y)
female.d.test = xgb.DMatrix(data=female.final.test.x,label=female.final.test.y)
### xgb tune
#xgboost grid
xgb.grid = expand.grid(eta = c(0.01,0.001),lambda = c(0.1,0.05),
                       max_depth = c(6,12,18),min_child_weight = c(0.1,0.05),
                       gamma = c(0.3,0.5),subsample = c(0.7,1),colsample_bytree = c(0.7,1))
male.xgb.perf.mat = matrix(0,nrow(xgb.grid),2)
female.xgb.perf.mat = matrix(0,nrow(xgb.grid),2)
colnames(male.xgb.perf.mat) = c("iter","score")
colnames(female.xgb.perf.mat) = c("iter","score")
#grid search
for(i in 1:nrow(xgb.grid)){
  print(i)
  params.xgb<-list(objective = "binary:logistic",booster = "gbtree",
                   gamma = xgb.grid[i,"gamma"],eta = xgb.grid[i,"eta"], #default = 0.3
                   max_depth = xgb.grid[i,"max_depth"],min_child_weight = xgb.grid[i,"min_child_weight"], 
                   subsample = xgb.grid[i,"subsample"],colsample_bytree = xgb.grid[i,"colsample_bytree"],
                   lambda = xgb.grid[i,"lambda"])
  male.xgbcv <- xgb.cv(params = params.xgb, data = male.d.train,nrounds = 1000,folds = male.f.cvIndex,
                       print_every_n = 50,early_stopping_rounds = 50, maximize = F, verbose = FALSE)
  female.xgbcv <- xgb.cv(params = params.xgb, data = female.d.train,nrounds = 1000,folds = female.f.cvIndex,
                         print_every_n = 50,early_stopping_rounds = 50, maximize = F, verbose = FALSE)
  male.xgb.perf.mat[i,]=c(male.xgbcv$best_iteration,min(male.xgbcv$evaluation_log$test_error_mean))
  female.xgb.perf.mat[i,]=c(female.xgbcv$best_iteration,min(female.xgbcv$evaluation_log$test_error_mean))
}

#find best tuning parameters
male.final.perf.xgb.mat = cbind(xgb.grid,male.xgb.perf.mat)
male.xgb.opt.par = male.final.perf.xgb.mat[which.min(male.final.perf.xgb.mat[,"score"]),]
female.final.perf.xgb.mat = cbind(xgb.grid,female.xgb.perf.mat)
female.xgb.opt.par = female.final.perf.xgb.mat[which.min(female.final.perf.xgb.mat[,"score"]),]

male.params.xgb<-list(objective = "binary:logistic",booster = "gbtree",
                      gamma = male.xgb.opt.par$gamma,eta = male.xgb.opt.par$eta, #default = 0.3
                      max_depth = male.xgb.opt.par$max_depth,min_child_weight = male.xgb.opt.par$min_child_weight,
                      subsample = male.xgb.opt.par$subsample,colsample_bytree = male.xgb.opt.par$colsample_bytree,
                      lambda = male.xgb.opt.par$lambda)
female.params.xgb<-list(objective = "binary:logistic",booster = "gbtree",
                        gamma = female.xgb.opt.par$gamma,eta = female.xgb.opt.par$eta, #default = 0.3
                        max_depth = female.xgb.opt.par$max_depth,min_child_weight = female.xgb.opt.par$min_child_weight,
                        subsample = female.xgb.opt.par$subsample,colsample_bytree = female.xgb.opt.par$colsample_bytree,
                        lambda = female.xgb.opt.par$lambda)

#final xgboost with optimal parameters
male.xgboost = xgb.train(data = male.d.train, params=male.params.xgb, nrounds = male.xgb.opt.par$iter)
female.xgboost = xgb.train(data = female.d.train, params=female.params.xgb, nrounds = female.xgb.opt.par$iter)

#final prediction
male.final.pred=(predict(male.xgboost,male.d.test)>0.5)+0
names(male.final.pred) = rownames(male.final.test.x)
female.final.pred=(predict(female.xgboost,female.d.test)>0.5)+0
names(female.final.pred) = rownames(female.final.test.x)

#submit
sex.xgb.pred=c(male.final.pred,female.final.pred)[rownames(final.test.x)]
final.submit=cbind(as.numeric(rownames(final.test.x)),sex.xgb.pred)
colnames(final.submit) = c("PassengerId","Survived")
write.csv(final.submit,paste(work.dir,"xgb.Sex.Fsize.submit.csv"),row.names = FALSE)


######################### lasso ##############################

foldid = rep(NA,length(male.final.train.y))
for(m in 1:folds){ foldid[male.f.cvIndex[[m]]] = m}
lasso.fit = cv.glmnet(male.final.train.x,male.final.train.y,family="binomial",foldid = foldid)
male.lasso.pred = (predict(lasso.fit,male.final.test.x,s=lasso.fit$lambda.min)>0)+0

foldid = rep(NA,length(female.final.train.y))
for(m in 1:folds){ foldid[female.f.cvIndex[[m]]] = m}
lasso.fit = cv.glmnet(female.final.train.x,female.final.train.y,family="binomial",foldid = foldid)
female.lasso.pred = (predict(lasso.fit,female.final.test.x,s=lasso.fit$lambda.min)>0)+0

sex.lasso.pred = rbind(male.lasso.pred,female.lasso.pred)[rownames(final.test.x),]
#final prediction
final.submit=cbind(as.numeric(rownames(final.test.x)),sex.lasso.pred)
colnames(final.submit) = c("PassengerId","Survived")
write.csv(final.submit,paste(work.dir,"lasso.Sex.Fsize.submit.csv"),row.names = FALSE)



######################### nn ##############################
num_epochs = 100
# 5 fold cross validation
cv.err.mat = NULL
for(i in 1:folds){
  cat("nn fold",i,"\n")
  val_data = as.matrix(male.final.train.x[male.f.cvIndex[[i]],]) ; partial_train_data = as.matrix(male.final.train.x[-male.f.cvIndex[[i]],])
  val_target = male.final.train.y[male.f.cvIndex[[i]]] ; partial_train_target = male.final.train.y[-male.f.cvIndex[[i]]]
  #build model
  model = keras_model_sequential() %>% 
    layer_dense(units = 64,activation = "relu", input_shape = dim(male.final.train.x)[2]) %>% #demension
    layer_dense(unit = 16,activation = "relu") %>% 
    layer_dense(unit = 8,activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")
  #compile model
  model %>% compile(optimizer = optimizer_adam(lr=0.001),loss = "binary_crossentropy",metrics = c("accuracy"))
  #save each epochs accuracy
  history = model %>% fit(partial_train_data,partial_train_target,
                          validation_data = list(val_data,val_target), #val data
                          epochs = num_epochs, batchsize = 1, verbose = 0)
  cv.err.mat = rbind(cv.err.mat,history$metrics$val_acc)
}
opt.epochs = which.max(colMeans(cv.err.mat))

###build model with optmal parameters
model = keras_model_sequential() %>% 
  layer_dense(units = 64,activation = "relu", input_shape = dim(male.final.train.x)[2]) %>% #demension
  layer_dense(unit = 16,activation = "relu") %>% 
  layer_dense(unit = 8,activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
###compile model
model %>% compile(optimizer = optimizer_adam(lr=0.001),loss = "binary_crossentropy",metrics = c("accuracy")
)

### final NN model with optimal tuning parameters
model %>% fit(as.matrix(male.final.train.x),male.final.train.y, #val data
              epochs = opt.epochs, batchsize = 1, verbose = 0)

male.nn.pred = (predict(model,as.matrix(male.final.test.x))>0.5)+0
rownames(male.nn.pred) = rownames(male.final.test.x)



num_epochs = 100
# 5 fold cross validation
cv.err.mat = NULL
for(i in 1:folds){
  cat("nn fold",i,"\n")
  val_data = as.matrix(female.final.train.x[female.f.cvIndex[[i]],]) ; partial_train_data = as.matrix(female.final.train.x[-female.f.cvIndex[[i]],])
  val_target = female.final.train.y[female.f.cvIndex[[i]]] ; partial_train_target = female.final.train.y[-female.f.cvIndex[[i]]]
  #build model
  model = keras_model_sequential() %>% 
    layer_dense(units = 64,activation = "relu", input_shape = dim(female.final.train.x)[2]) %>% #demension
    layer_dense(unit = 16,activation = "relu") %>% 
    layer_dense(unit = 8,activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")
  #compile model
  model %>% compile(optimizer = optimizer_adam(lr=0.001),loss = "binary_crossentropy",metrics = c("accuracy"))
  #save each epochs accuracy
  history = model %>% fit(partial_train_data,partial_train_target,
                          validation_data = list(val_data,val_target), #val data
                          epochs = num_epochs, batchsize = 1, verbose = 0)
  cv.err.mat = rbind(cv.err.mat,history$metrics$val_acc)
}
opt.epochs = which.max(colMeans(cv.err.mat))

###build model with optmal parameters
model = keras_model_sequential() %>% 
  layer_dense(units = 64,activation = "relu", input_shape = dim(female.final.train.x)[2]) %>% #demension
  layer_dense(unit = 16,activation = "relu") %>% 
  layer_dense(unit = 8,activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
###compile model
model %>% compile(optimizer = optimizer_adam(lr=0.001),loss = "binary_crossentropy",metrics = c("accuracy")
)

### final NN model with optimal tuning parameters
model %>% fit(as.matrix(female.final.train.x),female.final.train.y, #val data
              epochs = opt.epochs, batchsize = 1, verbose = 0)

female.nn.pred = (predict(model,as.matrix(female.final.test.x))>0.5)+0
rownames(female.nn.pred) = rownames(female.final.test.x)


sex.nn.pred = rbind(male.nn.pred,female.nn.pred)[rownames(final.test.x),]
#final prediction
final.submit=cbind(as.numeric(rownames(final.test.x)),sex.nn.pred)
colnames(final.submit) = c("PassengerId","Survived")
write.csv(final.submit,paste(work.dir,"nn.Sex.Fsize.submit.csv"),row.names = FALSE)


####################### Ensemble ###########################

sex.ens.pred=(apply(cbind(sex.xgb.pred,sex.svm.pred,sex.lasso.pred,sex.nn.pred,sex.RF.pred),1,sum)>=3)+0
final.submit=cbind(as.numeric(rownames(final.test.x)),sex.ens.pred)
colnames(final.submit) = c("PassengerId","Survived")
#submit
write.csv(final.submit,paste(work.dir,"sex.ensemble.Fsize.submit.csv"),row.names = FALSE)


save.image(paste(work.dir,"titanic_random1.RData",sep = ""))