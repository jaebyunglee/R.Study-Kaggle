  library(xgboost)
  library(glmnet)
  library(ncvreg)
  library(caret)
  library(data.table)
  library(plyr)
  library(psych)
  library(xlsx)
  library(e1071)
  library(randomForest)
  library(spatstat) #make dummy variables
  rm(list=ls())

  ######################################################################################
  train.data = read.csv("C:/Users/User/Desktop/jb/train.csv")
  train.data = train.data[,-1]
  test.data = read.csv("C:/Users/User/Desktop/jb/test.csv")
  test.data = test.data[,-1]
  test.data = data.frame(NA,test.data)
  colnames(test.data) = colnames(train.data)
  data = rbind(train.data,test.data)
  data = subset(data,select = -c(Name,Ticket))
  data$Pclass = as.factor(data$Pclass)
  levels(data$Cabin) = c(levels(data$Cabin),"None")
  data$Cabin[data$Cabin==""] = "None"
  data$Cabin = as.factor(substr(data$Cabin,1,1))
  levels(data$Embarked) = c(levels(data$Embarked),"None")
  data$Embarked[data$Embarked==""] = "None"
  data[is.na(data$Fare),]$Fare = mean(data[!is.na(data$Fare),]$Fare)
  
  
  #NA Age predict
  age.train = data[!is.na(data$Age),-1]
  age.test = data[is.na(data$Age),-1]
  age.fit = lm(Age~.,data=age.train)
  data[is.na(data$Age),]$Age = predict(age.fit,newdata = age.test)
  
  #scale numeric variables
  #data[,c("Age","SibSp","Parch","Fare")] = apply(data[,c("Age","SibSp","Parch","Fare")],2,scale)
  
  #train, test data
  train = data[!is.na(data$Survived),-8]
  test = data[is.na(data$Survived),-8]
  
  #train, valid data
  set.seed(2019)
  val.id = createDataPartition(train[,1],p=0.3,list=TRUE)
  valid = train[val.id$Resample1,]
  train = train[-val.id$Resample1,]
  
  #compare matrix
  com.mat = matrix(NA,3,15)
  #colnames(com.mat) = c("logit")
  rownames(com.mat) = c("acc","sen","spc")
  ############################## logit regression #############################
  glm.fit = glm(Survived~.,data=train,family = "binomial")
  tab = table(valid[,1],(predict(glm.fit,newdata = valid)>0)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,1] = c(acc,sen,spc)
  
  ############################## logit regression with backword#############################
  nullmod = glm(Survived~1,data=train,family = "binomial")
  fullmod = glm(Survived~.,data=train,family = "binomial")
  step.back = step(fullmod,, direction="backward") 
  tab = table(valid[,1],(predict(step.back,valid)>0)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,2] = c(acc,sen,spc)
  
  
  ############################# logit with lasso #############################
  train.mat = dummify(train)
  train.mat = subset(train.mat,select=-c(Pclass.3,Embarked.S))
  valid.mat = dummify(valid)
  valid.mat = subset(valid.mat,select=-c(Pclass.3,Embarked.S))
  lasso.fit = cv.glmnet(train.mat[,-1],train.mat[,1],family="binomial")
  tab = table(valid[,1],(predict(lasso.fit,newx = valid.mat[,-1])>0)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,3] = c(acc,sen,spc)
  
  ############################# logit with ridge #############################
  ridge.fit = cv.glmnet(train.mat[,-1],train.mat[,1],family="binomial",alpha=0)
  tab = table(valid[,1],(predict(ridge.fit,newx = valid.mat[,-1])>0)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,4] = c(acc,sen,spc)
  
  ############################ logit with scad ###############################
  scad.fit = cv.ncvreg(train.mat[,-1],train.mat[,1],family="binomial")
  tab = table(valid[,1],(predict(scad.fit, valid.mat[,-1])>0)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,5] = c(acc,sen,spc)
  
  ############################ logit with MCP ###############################
  MCP.fit = cv.ncvreg(train.mat[,-1],train.mat[,1],family="binomial",penalty="MCP")
  tab = table(valid[,1],(predict(MCP.fit, valid.mat[,-1])>0)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,6] = c(acc,sen,spc)
  
  ############################ decision tree ###############################
  set.seed(2019)
  tune_control = trainControl(method = "cv",number = 5 ,allowParallel = TRUE)
  tune_grid_dt = expand.grid(cp = seq(0,1,0.1))  
  dt_tune <- caret::train(x = train[,-1], y = as.factor(train[,1]),trControl = tune_control,
                          tuneGrid = tune_grid_dt, method = "rpart")  
  dt_clf = rpart(Survived~.,data = train,control = rpart.control(cp = 0),method="class")
  tab = table(valid[,1],(predict(dt_clf,newdata = valid)[,2]>0.5)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,7] = c(acc,sen,spc)
  
  ############################################Random Forest#########################################################
  tune.mat = NULL
  for(i in 1:100){
    print(i)
    set.seed(2019)
    tune_grid_rf = expand.grid(mtry = seq(1,7,1))  
    system.time(rf_tune <- caret::train(x = train[,-1], y = as.factor(train[,1]),
                                        trControl = tune_control,
                                        tuneGrid = tune_grid_rf,
                                        method = "rf",
                                        ntree = i,
                                        nthread = 12,
                                        verbose = TRUE
    ))
    opt.mtry  = which.max(rf_tune$results$Accuracy)
    opt.acc = rf_tune$results$Accuracy[opt.mtry]
    tune.mat = rbind(tune.mat,cbind(i,opt.mtry,opt.acc))
  }
  pos = which.max(tune.mat[,3])
  opt.vec = tune.mat[pos,]
  set.seed(2019)
  rf_clf = randomForest(x = train[,-1],y = as.factor(train[,1]),importance=TRUE,mtry=opt.vec[2],ntree = opt.vec[1])
  tab = table(valid[,1],(predict(rf_clf,valid)==1)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,8] = c(acc,sen,spc)
  
  ############################################ SVM #########################################################
  svm.train = dummify(train)
  svm.valid = dummify(valid)
  set.seed(2019)
  tune.svm = tune.svm(x=svm.train[,-1],y=as.factor(svm.train[,1]),gamma=2^(-5:5),cost=2^(-5:5),type="C")
  opt = tune.svm$best.parameters
  svm.fit = svm(x=svm.train[,-1],y=as.factor(svm.train[,1]),gamma=opt[1],cost=opt[2],type="C")
  tab = table(svm.valid[,1],predict(svm.fit,svm.valid[,-1]))
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,9] = c(acc,sen,spc)
  
  ################################xgboost did not tuning###################################
  sp.tr.data = sparse.model.matrix(Survived ~ .-1, data = train)
  sp.val.data = sparse.model.matrix(Survived ~ .-1, data = valid)
  tr.label = train[,1]
  val.label = valid[,1]
  d.train = xgb.DMatrix(data=sp.tr.data,label=tr.label)
  d.valid= xgb.DMatrix(data=sp.val.data,label=val.label)
  
  ## find optimal iter
  set.seed(2019)
  xgb.cv=xgb.cv(data = d.train, nrounds = 500, nthread = 4,
                nfold = 5,eta = 0.05,subsample=0.6,metrics=c("rmse","auc"),verbose=0)
  
  opt = which.min(xgb.cv$evaluation_log$test_rmse_mean)
  
  set.seed(2019)
  xgb.fit = xgboost(data=d.train,nrounds = opt,nthread = 4,eta = 0.05,subsample=0.6)
  tab = table(val.label,(predict(xgb.fit,sp.val.data)>0.5)+0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,10] = c(acc,sen,spc)
  
  ################################xgboost with tuning#################################
  ###input data
  input_y = as.factor(tr.label)
  input_x = as.matrix(sp.tr.data)
  
  ###stratified 5 fold
  set.seed(2019)
  folds <- 5
  cvIndex <- createFolds(factor(input_y), folds, returnTrain = F)
  
  ### tune1 nrounds, eta, max_depth
  tune_grid1 = expand.grid(nrounds = seq(200,500,50), 
                           eta = c(0.05,0.1,0.2,0.3),
                           max_depth = c(2,3,4,5,6),
                           gamma = 0,
                           colsample_bytree = 1,
                           min_child_weight = 1,
                           subsample = 1)    
  
  tune_control = trainControl(method = "cv", 
                              number = folds,
                              index = cvIndex ,
                              allowParallel = TRUE)
  set.seed(2019)
  system.time(xgb_tune <- caret::train(
    x = input_x,      #input_x is matrix
    y = input_y,      #input_y is factor
    trControl = tune_control,
    tuneGrid = tune_grid1,
    method = "xgbTree",
    nthread = 12,
    verbose = TRUE))
  
  max(xgb_tune$results$Accuracy)
  
  
  ###tune2 colsam,min_child_weight,subsample
  tune_grid2 = expand.grid(nrounds = xgb_tune$bestTune$nrounds, 
                           eta = xgb_tune$bestTune$eta,
                           max_depth = xgb_tune$bestTune$max_depth,
                           gamma = 0,
                           colsample_bytree = c(0.6,0.7,0.8,1),
                           min_child_weight = c(1,3,5),
                           subsample = c(0.5,0.75,1))       
  set.seed(2019)
  system.time(xgb_tune2 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid2,
    method = "xgbTree",
    nthread = 12,
    verbose = TRUE))
  
  max(xgb_tune2$results$Accuracy)
  
  ###tune3 gamma
  tune_grid3 = expand.grid(nrounds = xgb_tune2$bestTune$nrounds, 
                           eta = xgb_tune2$bestTune$eta,
                           max_depth = xgb_tune2$bestTune$max_depth,
                           gamma = c(0,1,3,5),
                           colsample_bytree = xgb_tune2$bestTune$colsample_bytree,
                           min_child_weight = xgb_tune2$bestTune$min_child_weight,
                           subsample = xgb_tune2$bestTune$subsample)    
  set.seed(2019)
  system.time(xgb_tune3 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid3,
    method = "xgbTree",
    nthread = 12,
    verbose = TRUE))
  
  max(xgb_tune3$results$Accuracy)
  
  # ###tun4 nrounds eta
  # tune_grid4 = expand.grid(nrounds = seq(100,4000,500),
  #                          eta = seq(0.05,0.07,0.02),
  #                          max_depth = xgb_tune3$bestTune$max_depth,
  #                          gamma = xgb_tune3$bestTune$gamma,
  #                          colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  #                          min_child_weight = xgb_tune3$bestTune$min_child_weight,
  #                          subsample = xgb_tune3$bestTune$subsample)
  # set.seed(2019)
  # system.time(xgb_tune4 <- caret::train(
  #   x = input_x,
  #   y = input_y,
  #   trControl = tune_control,
  #   tuneGrid = tune_grid4,
  #   method = "xgbTree",
  #   nthread = 12,
  #   verbose = TRUE))
  # max(xgb_tune4$results$Accuracy)
  # 
  ###find best iteration
  default_param<-list(objective = "binary:logistic",
                      booster = "gbtree",
                      eta = xgb_tune3$bestTune$eta, #default = 0.3
                      gamma = xgb_tune3$bestTune$gamma,
                      max_depth = xgb_tune3$bestTune$max_depth, #default=6
                      min_child_weight = xgb_tune3$bestTune$min_child_weight, #default=1
                      subsample = xgb_tune3$bestTune$subsample,
                      colsample_bytree = xgb_tune3$bestTune$colsample_bytree)
  
  
  
  set.seed(2019)
  xgbcv <- xgb.cv(params = default_param, data = d.train,
                  nrounds = 10000,
                  folds = cvIndex, print_every_n = 200, 
                  early_stopping_rounds = 2000, maximize = F)
  
  best.iter = which.min(xgbcv$evaluation_log$test_error_mean)
  ###final model
  xgb_mod <- xgb.train(data = d.train, params=default_param, nrounds = best.iter)
  tab = table(val.label,(predict(xgb_mod,sp.val.data)>0.5) + 0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,11] = c(acc,sen,spc)
  
  
  
  
  ###########################################################################################
  
  # f.mat = NULL
  # colnames(f.mat) = c("eta","gam","depth","weight","sbsam","clsam")
  # for(eta in c(0.03,0.05,0.1,0.3)){
  #   for(gam in c(1,2,3)){
  #     for(dep in c(2,4,6,8)){
  #       for(wei in c(1,2,3)){
  #         for (sbsam in c(0.6,0.8,1)){
  #           for (clsam in c(0.6,0.8,1)){
  #             print(eta)
  #             default_param<-list(objective = "binary:logistic",
  #                                 booster = "gbtree",
  #                                 eta = eta, #default = 0.3
  #                                 gamma = gam,
  #                                 max_depth = dep, #default=6
  #                                 min_child_weight = wei, #default=1
  #                                 subsample = sbsam,
  #                                 colsample_bytree = clsam)
  #             
  #             set.seed(2019)
  #             xgbcv <- xgb.cv(params = default_param, data = d.train,
  #                             nrounds = 5000,
  #                             folds = cvIndex, print_every_n = 200, 
  #                             early_stopping_rounds = 500, maximize = F)
  #             best.iter = which.min(xgbcv$evaluation_log$test_error_mean)
  #             sacc = min(xgbcv$evaluation_log$test_error_mean)
  #             f.mat = rbind(f.mat,c(eta,gam,dep,wei,sbsam,clsam,best.iter,sacc))
  #           }
  #         }
  #       }
  #     }
  #   }
  # }
  # 
 
  oopt = which.min(f.mat[,8])
  f.mat[oopt,] 
  #best tuning parameters using grid search
  default_param<-list(objective = "binary:logistic",
                      booster = "gbtree",
                      eta = 0.3, #default = 0.3
                      gamma = 2,
                      max_depth = 8, #default=6
                      min_child_weight = 2, #default=1
                      subsample = 0.6,
                      colsample_bytree = 1)
  
  
  
  
  set.seed(2019)
  xgbcv <- xgb.cv(params = default_param, data = d.train,
                  nrounds = 10000,
                  folds = cvIndex, print_every_n = 200, 
                  early_stopping_rounds = 2000, maximize = F)
  
  best.iter = which.min(xgbcv$evaluation_log$test_error_mean)
  ###final model
  xgb_mod <- xgb.train(data = d.train, params=default_param, nrounds = best.iter)
  tab = table(val.label,(predict(xgb_mod,sp.val.data)>0.5) + 0)
  acc = sum(diag(tab))/sum(tab)
  sen = tab[2,2]/sum(tab[2,])
  spc = tab[1,1]/sum(tab[1,])
  com.mat[,11] = c(acc,sen,spc)
