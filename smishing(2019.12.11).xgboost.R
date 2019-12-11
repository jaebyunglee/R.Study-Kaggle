rm(list=ls())
library(quanteda)
library(RColorBrewer)
library(stringr)
library(ggplot2)
library(dplyr)
library(grid)
library(gridExtra)
library(caret)
library(pROC)
library(randomForest)
library(lightgbm)
library(ggpubr)
library(text2vec)
library(glmnet)
library(xgboost)
#원본데이터
or.train=read.table("C:\\Users\\User\\Desktop\\jb\\filedown\\train.csv",sep=",",encoding = "UTF-8",fill=T,header=T)
or.test=read.table("C:\\Users\\User\\Desktop\\jb\\filedown\\public_test.csv",sep=",",fill=T,header=T)



#################################################################################################################
DATA = "data1" 
#data1: 아무것도 변환하지 않은것, data2: 1~4월데이터, data3: 서브샘플링 
#data4 : data2 & data3, data5 : 2018년 데이터, data6 : 2018이면서 01~04월인 데이터
train = or.train
test = or.test


#text변수를 chracter로 변환
train$text=as.character(train$text)
test$text=as.character(test$text)
#스미싱 여부가 NA인 자료의 경우 text변수의 마지막 글자에 스미싱 여부가 포함되어 있음
#스미싱이 NA인 데이터의 텍스트 추출
xx=train[is.na(train$smishing),"text"]
#텍스트의 맨 뒤 글자에 있는 스미싱 여부 추출
aa=str_sub(xx,-1)
table(aa)
#테스트의 맨 뒤 글자에 있는 콤마와 스미싱 여부 텍스트 제거
str_sub(xx,-2) = ""

#콤마와 스미싱 여부가 제거된 텍스트를 텍스트데 변수의 데이터로 사용
train[is.na(train$smishing),"text"] = xx

#스미싱 여부가 NA인 데이터 채워넣음
train[is.na(train$smishing),"smishing"] = aa
table(train$smishing)

#smishing 변수 숫자로 변환
train$smishing = as.numeric(train$smishing)


#####################################################################
# if(DATA=="data2"){
#   #학습 데이터중 01~04월 데이터만 사용
#   month.id=substr(train$year_month,6,7) %in% c("01","02","03","04")
#   train = train[month.id,]
# }
# 
# if(DATA=="data3"){
#   #리샘플링 하기
#   re.smi=train[train$smishing==1,]
#   dd=dim(train[train$smishing==0,])
#   re.n.smi=train[train$smishing==0,][sample(1:dd[1],dd[1]*0.7),]
#   train = rbind(re.smi,re.n.smi)
# }
# 
# if(DATA=="data4"){
#   month.id=substr(train$year_month,6,7) %in% c("01","02","03","04")
#   train = train[month.id,]
#   re.smi=train[train$smishing==1,]
#   re.n.smi=train[train$smishing==0,][sample(1:dim(train[train$smishing==0,])[1],dim(train[train$smishing==0,])[1]*0.7),]
#   train = rbind(re.smi,re.n.smi)
# }
# if(DATA=="data5"){
#   train=train[substr(train$year_month,3,4)=="18",]
# }
# if(DATA=="data6"){
#   train=train[substr(train$year_month,3,4)=="18" & substr(train$year_month,6,7)%in%c("01","02","03","04"),]
# }

#####################################################################
set.seed(2019)

#token
train_it <- itoken(train$text, tokenizer=word_tokenizer)
test_it  <- itoken(test$text,  tokenizer=word_tokenizer)


#특수문자 변수 제거 & 길이가 1인 문자 제거
del.vec = c(".",")","(","!","-",":","%","?","XXX","XXX-XXX-XXX",",","0","XX-XXX-XXX","-XXX","_"," ","","  ","    ","     ","      ","       ") #특수문자
del.vec = c(del.vec,as.vector(outer(c(0:9),c(0:9),paste,sep=""))) # "00","01","02"~"99" 등의 문자
del.vec = c(del.vec,c("이라고","있습니다","드립니다","바랍니다","합니다","되세요","또한","입니다","또는")) #기타문자


#Create vocab and prune
vocab <- create_vocabulary(train_it,ngram = c(ngram_min = 1L, ngram_max = 1L ),stopwords = del.vec)
pruned_vocab = prune_vocabulary(vocab,doc_proportion_min=0.001)

#Create dtm
train.dtm <- create_dtm(train_it, vocab_vectorizer(pruned_vocab))
test.dtm  <- create_dtm(test_it,  vocab_vectorizer(pruned_vocab))

# TF-IDF
tfidf <- TfIdf$new()
train_tfidf <- tfidf$fit_transform(train.dtm)
test_tfidf  <- tfidf$transform(test.dtm)


z = 30
#테스트 데이터 tfidf 기준 상위 z개
test.bb=sort(colSums(test_tfidf),decreasing = T)[1:z]
#학습 데이터 tfidf 기준 상위 z개
train.bb = sort(colSums(train_tfidf),decreasing = T)[1:z]
#학습 데이터 중 스미싱 아닌 데이터 tfidf 기준 상위 z개
train.nn.bb = sort(colSums(train_tfidf[train$smishing==0,]),decreasing = T)[1:z]
#학습 데이터 중 스미싱 데이터 tfidf 기준 상위 z개
train.pp.bb = sort(colSums(train_tfidf[train$smishing==1,]),decreasing = T)[1:z]


#테스트데이터의 문자열 'tfidf' 그래프
g1=ggplot() + geom_bar(aes(x=reorder(names(test.bb),-test.bb/dim(test.dtm)[1]),y=test.bb/dim(test.dtm)[1]), stat='identity',fill="black",color="gray")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  labs(x="",y = 'tfidf')

#학습데이터의 문자열 'tfidf' 그래프
g2=ggplot() + geom_bar(aes(x=reorder(names(train.bb),-train.bb/dim(train.dtm)[1]),y=train.bb/dim(train.dtm)[1]), stat='identity',fill="pink",color="red")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  labs(x="",y = 'tfidf')

#학습데이터 중 스미싱데이터 문자열 'tfidf' 그래프
g3=ggplot() + geom_bar(aes(x=reorder(names(train.pp.bb),-train.pp.bb/sum(train$smishing==1)),y=train.pp.bb/sum(train$smishing==1)), stat='identity',fill="skyblue",color="blue")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  labs(x="",y = 'tfidf')

#학습데이터 중 스미싱아닌 데이터 문자열 'tfidf' 그래프
g4=ggplot() + geom_bar(aes(x=reorder(names(train.nn.bb),-train.nn.bb/sum(train$smishing==0)),y=train.nn.bb/sum(train$smishing==0)), stat='identity',fill="gray",color="darkgray")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  labs(x="",y = 'tfidf')


#g1, g2,g3,g4 그래프
ggarrange(ggarrange(g1, g2, ncol = 2, labels = c("Test data", "Train data")), # First row with scatter plot
          ggarrange(g3, g4, ncol = 2, labels = c("smi = 1", " smi = 0 ")), # Second row with box and dot plots
          nrow = 2)                                    # Labels of the scatter plot


#smishing wordcloud
train.wordcloud = as.dfm(train.dtm)
smishing.col = brewer.pal(10, "BrBG")  
smishing.cloud = textplot_wordcloud(train.wordcloud[train$smishing==1,], min_count = 3000, color = smishing.col)  
title("Smishing Wordcloud", col.main = "grey14")

#not smishing wordcloud
smishing.col = brewer.pal(10, "BrBG")  
smishing.cloud = textplot_wordcloud(train.wordcloud[train$smishing==0,], min_count = 7000, color = smishing.col)  
title("Not smishing Wordcloud", col.main = "grey14")

#학습 데이터 월별 문자수 건수, 스미싱 건수 그래프

g5=ggplot(train[,c(2,4)], aes(x = year_month, fill = as.factor(smishing))) +
  geom_bar(stat='count', position='dodge') + theme_grey() +
  labs(x = 'Training data only') +
  geom_label(stat='count', aes(label=..count..))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


g6=ggplot(train[,c(2,4)], aes(x = year_month, fill = as.factor(smishing))) +
  geom_bar( position='fill') +
  labs(x = 'Training data only', y = "Percent") + scale_y_continuous(labels=scales::percent)+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


ggarrange(g5,g6,nrow = 2, labels =c("Count","Percent"))

# feature hashing
train_hash <- create_dtm(train_it, hash_vectorizer(hash_size=2^12))
test_hash  <- create_dtm(test_it,  hash_vectorizer(hash_size=2^12))

# final train and test data
final_train <- cbind(train_tfidf, train_hash)
final_test <- cbind(test_tfidf, test_hash)




############################ Model ##################################



# train & valid data
id = createDataPartition(y=train$smishing,p=0.7,list=F)
partial_train = final_train[as.vector(id),]
partial_valid = final_train[-as.vector(id),]
partial_train_label = train$smishing[as.vector(id)]
partial_valid_label = train$smishing[-as.vector(id)]


###### lightgbm #####
#light gbm data
lgb.train = lgb.Dataset(data = final_train, label = train$smishing)
lgb.p.train = lgb.Dataset(data = partial_train ,label=partial_train_label)
lgb.p.valid = lgb.Dataset.create.valid(lgb.p.train, data = partial_valid ,label=partial_valid_label)

#tune grid
lgb.grid =  expand.grid(num_leaves = c(2^6), #xgboost max_depth, light gbm is leaf wise
                        learning_rate = c(0.1), #xgboost eta
                        feature_fraction = 0.8, # xgboost subsample
                        bagging_fraction = 0.8,
                        lambda_l1 = c(0.1,0.01,0.001))

#performance matrix
lgb.perf.mat = matrix(0,nrow(lgb.grid),2)
colnames(lgb.perf.mat) = c("iter","score")

#grid search
for(i in 1:nrow(lgb.grid)){
  cat("lightgbm iteration",i,"\n")
  params.lgb = list(
    objective = "binary"
    , metric = "auc"
    , learning_rate = lgb.grid[i,"learning_rate"]
    , num_leaves = lgb.grid[i,"num_leaves"]
    , min_sum_hessian_in_lear = lgb.grid[i,"min_sum_hessian_in_lear"]
    , feature_fraction = lgb.grid[i,"feature_fraction"]
    , bagging_fraction = lgb.grid[i,"bagging_fraction"]
    , lambda_l1 = lgb.grid[i,"lambda_l1"]
  )
  
  lgb.model = lgb.train(
    params = params.lgb
    , data = lgb.p.train
    , valid = list(test=lgb.p.valid)
    , num_threads = 12
    , nrounds = 1000
    , early_stopping_rounds = 50
    , verbose = -1
    , boost_from_average=FALSE
  )
  
  lgb.perf.mat[i,] = c(lgb.model$best_iter,lgb.model$best_score)
  
}

#optimal parameters
final.lgb.perf.mat = cbind(lgb.grid,lgb.perf.mat)
lgb.opt.par = final.lgb.perf.mat[which.max(final.lgb.perf.mat$score),]
cat("light gbm optimal parameters : ","\n",paste(colnames(lgb.opt.par),lgb.opt.par,sep=":",collapse = "  "),"\n")

final.params.lgb = list(
  objective = "binary"
  , metric = "auc"
  , learning_rate = lgb.opt.par$learning_rate
  , num_leaves = lgb.opt.par$num_leaves
  , feature_fraction = lgb.opt.par$feature_fraction
  , bagging_fraction = lgb.opt.par$bagging_fraction
  , lambda_l1 = lgb.opt.par$lambda_l1
)


#final lightgbm model
final.lgb.model = lgb.train(params = final.params.lgb, data = lgb.train,
                            num_threads=12, nrounds = lgb.model$best_iter)




##### xgboost #####
#xgboost data
xgb.train = xgb.DMatrix(data=final_train,label=train$smishing)
xgb.p.train = xgb.DMatrix(data=partial_train,label=partial_train_label)
xgb.p.valid = xgb.DMatrix(data=partial_valid,label=partial_valid_label)

#xgboost grid
xgb.grid = expand.grid(eta = c(0.1),
                       lambda = c(0.1,0.01,0.001),
                       max_depth = c(6),
                       min_child_weight = c(0.1,0.01),
                       gamma = c(0.3),
                       subsample = 0.8,
                       colsample_bytree = 0.8)
xgb.perf.mat = matrix(0,nrow(xgb.grid),2)
colnames(xgb.perf.mat) = c("iter","score")

#grid search
for(i in 1:nrow(xgb.grid)){
  cat("xgboost iteration",i,"\n")
  params.xgb<-list(objective = "binary:logistic",
                   booster = "gbtree",
                   gamma = xgb.grid[i,"gamma"],
                   eta = xgb.grid[i,"eta"], #default = 0.3
                   max_depth = xgb.grid[i,"max_depth"], #default=6
                   min_child_weight = xgb.grid[i,"min_child_weight"], #default=1
                   subsample = xgb.grid[i,"subsample"],
                   colsample_bytree = xgb.grid[i,"colsample_bytree"],
                   lambda = xgb.grid[i,"lambda"])
  
  xgb.model = xgb.train(params = params.xgb
                        , data = xgb.p.train
                        , watchlist = list(val=xgb.p.valid)
                        , nround = 1000
                        , early_stopping_rounds = 50
                        , eval_metric = "auc"
                        , maximize = T
                        , print_every_n = 20
  )
  
  xgb.perf.mat[i,] = c(xgb.model$best_iteration,xgb.model$best_score)
}

#optimal parameters
final.xgb.perf.mat = cbind(xgb.grid,xgb.perf.mat)
xgb.opt.par = final.xgb.perf.mat[which.max(final.xgb.perf.mat$score),]
cat("xgboost optimal parameters : ","\n",paste(colnames(xgb.opt.par),xgb.opt.par,sep=":",collapse = "  "),"\n")
final.params.xgb<-list(objective = "binary:logistic",
                 booster = "gbtree",
                 gamma = xgb.opt.par$gamma,
                 eta = xgb.opt.par$eta, #default = 0.3
                 max_depth = xgb.opt.par$max_depth, #default=6
                 min_child_weight = xgb.opt.par$min_child_weight, #default=1
                 subsample = xgb.opt.par$subsample,
                 colsample_bytree = xgb.opt.par$colsample_bytree,
                 lambda = xgb.opt.par$lambda)

#final xgboost model
final.xgb.model = xgb.train(params = final.params.xgb, data = xgb.train, nrounds = xgb.opt.par$iter)



#test prediction
lgb.pred = predict(final.lgb.model,final_test)
xgb.pred = predict(final.xgb.model,final_test)


#결과 값 저장
lgb.result = data.frame(id = test$id,smishing=lgb.pred) #lightgbm
xgb.result = data.frame(id = test$id,smishing=xgb.pred) #xgboost
write.csv(lgb.result,"C:\\Users\\User\\Desktop\\jb\\filedown\\result.lgb.2019.12.11.csv",row.names = F)
write.csv(xgb.result,"C:\\Users\\User\\Desktop\\jb\\filedown\\result.xgb.2019.12.11.csv",row.names = F)
