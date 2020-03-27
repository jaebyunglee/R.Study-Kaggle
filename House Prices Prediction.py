# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:41:00 2020

@author: begas
"""

# -------------------------------------------------------------------
# 파일명 : House Prices Prediction.py
# 설  명 : 파이썬을 활용한 집값 예측
# 작성자 : 이재병(010-2775-0930, jblee@begas.co.kr)
# 작성일 : 2020/03/20
# 패키지 : numpy, pandas ......
# ggplot 파이썬 참고자료 : https://monashdatafluency.github.io/python-workshop-base/modules/plotting_with_ggplot/
#                       : https://plotnine.readthedocs.io/en/stable/ (공식문서)
# --------------------------------------------------------------------
#%% 스파이더 업데이트 방법
# conda update conda
# conda update anaconda
# conda update spyder

#%% 패키지 불러오기

import numpy as np
import pandas as pd
import seaborn as sns # heatmap
import matplotlib.pyplot as plt #plot

from numpy import nan # NA로 강제 지정
from plotnine import * # 파이썬에서 ggplot2 사용, pip로 설치
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder #범주형 변수 인코딩

#Padas display option
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 80)

#%% 작업공간 지정하기
import os
os.getcwd()
os.chdir("C:/Pyproject")


#%% 데이터 불러오기
train = pd.read_csv("./DAT/train.csv")
test = pd.read_csv("./DAT/test.csv")
test_labels = test["Id"] #테스트 라벨 저장
train = train.drop("Id",1)
test = test.drop("Id",1)
test["SalePrice"] = nan
all = pd.concat([train,test], axis = 0) # 데이터 행으로 합치기

#%% 데이터 살펴보기

#전체 데이터 Missing value
na_num = all.isnull().sum()/all.shape[0]
na_num[na_num!=0]

# Train 데이터 Missing value
tr_na_num = train.isnull().sum()/train.shape[0]
tr_na_num[tr_na_num!=0]

# Test 데이터 Missing value
te_na_num = test.isnull().sum()/test.shape[0]
te_na_num[te_na_num!=0]

# Train 데이터에만 있는 Missing value 파악
tr_na_val_names = list(tr_na_num[tr_na_num!=0].index.values) #Train 중 NA 있는 병수명 추출
te_na_val_names = list(te_na_num[te_na_num!=0].index.values) #Test 중 NA 있는 병수명 추출

set(tr_na_val_names) - set(te_na_val_names) #Electrical 변수 Train에만 결측값 존재
all = all[-all["Electrical"].isnull()] #row 제거

#숫자형 변수명 추출
num_var_names = all.select_dtypes(include=np.number).columns # 숫자형 변수명 추출
num_var_names = num_var_names[num_var_names!="SalePrice"]
num_var_names = set(num_var_names) - set(["MSSubClass","OverallQual","OverallCond","BsmtFullBath","BsmtHalfBath","GarageCars"])
num_var_names = list(num_var_names)

len(num_var_names)

#문자형 변수명 추출
all_var_names = all.columns.values[all.columns.values!="SalePrice"]
cat_var_names = set(all_var_names) - set(num_var_names)
cat_var_names = list(cat_var_names)
len(cat_var_names)


#%% 결측값 탐색 및 처리하기

### 범주형 변수 결측값 Imputation

#Test에만 결측이 있는경우 -> 최빈값 대체
#결측치 수가 30개 미만 -> 최빈값 대체
#결측치수가 30개 이상 -> "N"이라는 새로운 범주로 대체
#결측치가 "None"이라는 범주의 의미 -> "None" 범주로 대체

#MSZoning 변수, 결측값 4개, 최빈값으로 대체
all["MSZoning"].isnull().sum() #결측값 4개
all["MSZoning"].value_counts() #최빈값 확인
all["MSZoning"] = all["MSZoning"].fillna(value = "RL")

#Alley 변수, NA가 골목 없음을 의미, NA를 "N" 범주로 대체
all["Alley"].isnull().sum()
all["Alley"] = all["Alley"].fillna(value="None")

#Utilities 변수, 결측값 2개, 최빈값으로 대체
all["Utilities"].isnull().sum()
Uti_mode = all["Utilities"].mode()
all["Utilities"] = all["Utilities"].fillna(value = Uti_mode[0])

#Exterior1st 변수, 결측값 1개, 최빈값으로 대체
all["Exterior1st"].isnull().sum()
Ext1_mode = all["Exterior1st"].mode()
all["Exterior1st"] = all["Exterior1st"].fillna(value = Ext1_mode[0])

#Exterior2nd 변수, 결측값 1개, 최빈값으로 대체
all["Exterior2nd"].isnull().sum()
Ext2_mode = all["Exterior2nd"].mode()
all["Exterior2nd"] = all["Exterior2nd"].fillna(value = Ext2_mode[0])

#MasVnrType 변수, 결측값 24개, 최빈값으로 대체
all["MasVnrType"].isnull().sum()
Mas_mode = all["MasVnrType"].mode()
all["MasVnrType"] = all["MasVnrType"].fillna(value = Mas_mode[0])

#BsmtQual 변수, 결측값 81개, NA가 None 의미
all["BsmtQual"].isnull().sum()
all["BsmtQual"] = all["BsmtQual"].fillna(value = "None")

#BsmtCond 변수, 결측값 82개, NA가 None 의미
all["BsmtCond"].isnull().sum()
all["BsmtCond"] = all["BsmtCond"].fillna(value = "None")

#BsmtExposure 변수, 결측값 82개, NA가 None 의미
all["BsmtExposure"].isnull().sum()
all["BsmtExposure"] = all["BsmtExposure"].fillna(value = "None")

#BsmtFinType1 변수, 결측값 79개, NA가 None 의미
all["BsmtFinType1"].isnull().sum()
all["BsmtFinType1"] = all["BsmtFinType1"].fillna(value = "None")

#BsmtFinType2 변수, 결측값 80개, NA가 None 의미 
all["BsmtFinType2"].isnull().sum()
all["BsmtFinType2"] = all["BsmtFinType2"].fillna(value = "None")

#BsmtFullBath 변수, 결측값 2개, 최빈값으로 대체
all["BsmtFullBath"].isnull().sum()
BsmFB_mode = all["BsmtFullBath"].mode()
all["BsmtFullBath"] = all["BsmtFullBath"].fillna(value = BsmFB_mode[0])

#BsmtHalfBath 변수, 결측값 2개, 최빈값으로 대체
all["BsmtHalfBath"].isnull().sum()
BsmHB_mode = all["BsmtHalfBath"].mode()
all["BsmtHalfBath"] = all["BsmtHalfBath"].fillna(value = BsmHB_mode[0])

#KitchenQual 변수, 결측값 1개, 최빈값으로 대체
all["KitchenQual"].isnull().sum()
Kit_mode = all["KitchenQual"].mode()
all["KitchenQual"] = all["KitchenQual"].fillna(value = Kit_mode[0])

#Functional 변수, 결측값 2개, 최빈값으로 대체
all["Functional"].isnull().sum()
Fun_mode = all["Functional"].mode()
all["Functional"] = all["Functional"].fillna(value = Fun_mode[0])

#FireplaceQu 변수, 결측값 1419개, NA가 None을 의미
all["FireplaceQu"].isnull().sum()
all["FireplaceQu"] = all["FireplaceQu"].fillna(value = "None")

#GarageType 변수, 결측값 157개, N 범주로 대체
all["GarageType"].isnull().sum()
all["GarageType"] = all["GarageType"].fillna(value = "N")

#GarageFinish 변수, 결측값 159개, N 범주로 대체
all["GarageFinish"].isnull().sum()
all["GarageFinish"] = all["GarageFinish"].fillna(value = "N")

#GarageCars 변수, 결측값 1개, 최빈값으로 대체
all["GarageCars"].isnull().sum()
GarC_mode = all["GarageCars"].mode()
all["GarageCars"] = all["GarageCars"].fillna(value = GarC_mode[0])

#GarageQual 변수, 결측값 159개, NA가 None 의미
all["GarageQual"].isnull().sum()
all["GarageQual"] = all["GarageQual"].fillna(value = "None")

#GarageCond 변수, 결측값 159개, NA가 None 의미
all["GarageCond"].isnull().sum()
all["GarageCond"] = all["GarageCond"].fillna(value = "None")

#PoolQC 변수, 결측값 2908개, NA가 None의미
all["PoolQC"].isnull().sum()
all["PoolQC"] = all["PoolQC"].fillna(value = "None")

#Fence 변수, 결측값 2347개, NA가 None 의미
all["Fence"].isnull().sum()
all["Fence"] = all["Fence"].fillna(value = "None")

#MiscFeature 변수, 결측값 2813개, NA가 None의미
all["MiscFeature"].isnull().sum()
all["MiscFeature"] = all["MiscFeature"].fillna(value = "None")

#SaleType 변수, 결측값 1개, 최빈값 대체
all["SaleType"].isnull().sum()
ST_mode = all["SaleType"].mode()
all["SaleType"] = all["SaleType"].fillna(value = ST_mode[0])


#범주형 변수 Category로 변환
for ind in cat_var_names:
    all[ind] = all[ind].astype("category")


# --------------------------------------------------------------

### 숫자형 변수 결측값 Imputation
#숫자형 변수 중 결측이 있는 변수 확인
num_na = all[num_var_names].isnull().sum()[all[num_var_names].isnull().sum()!=0]


# 결측값 30개 미만 Median으로 대체
num_na_low = list(num_na.index.values[num_na<30])
for ind in num_na_low:        
    n_mean = all[ind].mean()
    all[ind] = all[ind].fillna(value = n_mean) 


# 결측값 30개 이상인 경우 상관계수가 높은 10개 변수로 회귀분석하여 예측 대체
num_na_high = list(num_na.index.values[num_na>=30])
for ind in num_na_high:
    #상관계수가 높은 10개 변수 추출
    cor_mat = all[num_var_names].corr()
    temp_order = cor_mat.abs().sort_values(by = ind,axis=0, ascending=False).sort_values(by = ind,axis=1, ascending=False)
    high_cor_names = list(temp_order.iloc[1:11,1:11].index.values)
    
    # 회귀분석을 위한 데이터 준비
    train_x = all[high_cor_names][-all[ind].isnull()]
    train_y = all[ind][-all[ind].isnull()]
    test_x = all[high_cor_names][all[ind].isnull()]

    # 회귀분석 실행
    mlr = LinearRegression()
    mlr.fit(train_x,train_y)

    # 예측값으로 imputation
    all[ind][all[ind].isnull()] = mlr.predict(test_x)

all[["LotFrontage","GarageYrBlt"]].isnull().sum()

##%% 탐색적 자료분석
##############################################################
## 그래프 저장                                               #
## spyder 옵션병경 Tools - Preferences - IPython console     #
## Graphics - Backend를 Automatic 으로                       #
##############################################################
#
## 숫자형 독립변수들 간의 상관계수
#cor_mat = all[num_var_names].corr()
#plt.figure(figsize=(20,10)) #plot size
#sns_plot = sns.heatmap(data = cor_mat, annot=True, fmt = '.2f') #sns plot
#figure = sns_plot.get_figure() #get plot
#figure.savefig("OUT/num_var_cor.png") #save plot
#
## 종속변수와 관련이 높은 독립변수 상관계수
#cor_mat = all[num_var_names + ["SalePrice"]][-all["SalePrice"].isnull()].corr() #cormat
#high_cor_mat = cor_mat.abs().sort_index(by = "SalePrice", axis = 0, ascending = False).sort_index(by = "SalePrice", axis = 1, ascending = False) #cor 높은 순으로 인덱스 추출
#cor_mat = cor_mat.loc[list(high_cor_mat.index),list(high_cor_mat.index)] #cor_mat에서 cor 높은순으로 정렬
#plt.figure(figsize=(20,10)) #plot size
#sns_plot = sns.heatmap(cor_mat, annot=True, fmt=".2f") #sns plot
#figure = sns_plot.get_figure() #get plot
#figure.savefig("OUT/Y_num_var_cor.png") #save plot
#
## 종속변수 독립변수 Histogram, Boxplot
#p1 = ggplot(all[-all["SalePrice"].isnull()], aes(x="SalePrice")) + geom_histogram(colour="black",bins=30)
#p1 = p1 + labs(y = "Count") + ggtitle("Histogram fo SalePrice")
#p2 = ggplot(all[-all["SalePrice"].isnull()], aes(x="SalePrice",y="SalePrice")) + geom_boxplot(colour="black",fill="darkgrey")
#p2 = p2 + ggtitle("Boxplot of SalePrice")
#
## 그래프 저장
#p1.save("./OUT/EDA_Y_SalesPrice1.png",width = 7, height = 3)
#p2.save("./OUT/EDA_Y_SalesPrice2.png",width = 3, height = 3)
#
## 숫자형 독립변수 EDA
#for g_i in num_var_names:
#    p1 = ggplot(all[-all["SalePrice"].isnull()], aes(x=g_i)) + geom_histogram(colour="black",bins=30)
#    p1 = p1 + labs(y = "Count") +ggtitle("Histogram of "+ g_i) + theme(title = element_text(hjust = 0.5))
#    p2 = ggplot(all[-all["SalePrice"].isnull()], aes(x=g_i, y=g_i)) + geom_boxplot()
#    p2 = p2 + labs(x = " ") + ggtitle("Boxplot of " + g_i) + theme(title = element_text(hjust = 0.5))
#    p3 = ggplot(all[-all["SalePrice"].isnull()], aes(x=g_i, y="SalePrice")) + geom_point(colour="blue")
#    p3 = p3 + ggtitle("Scatter plot of "+ g_i + " and SalePrice") + theme(title = element_text(hjust = 0.5))
#    p3 = p3 + geom_smooth(method = "lm", color = "black")
#    p1.save("./OUT/EDA_HIST_" + g_i +".png", width = 7, height = 3)
#    p2.save("./OUT/EDA_BOX_" + g_i +".png", width = 7, height = 3)
#    p3.save("./OUT/EDA_SCAT_" + g_i +".png", width = 7, height = 3)
#
## 범주형 독립변수 EDA
#for g_i in cat_var_names:
#    p1 = ggplot(all[-all["SalePrice"].isnull()], aes(x = g_i, y = "SalePrice")) + geom_boxplot()
#    p1 = p1 + geom_hline(yintercept=all["SalePrice"].median(), linetype = "dashed", colour = "red")
#    p1 = p1 + ggtitle("Boxplot of SalePrice ~ " + g_i) + theme(title=element_text(hjust=0.5))
#    p1 = p1 + theme(axis_text_x = element_text(angle = 45, hjust = 1))
#    p1.save("./OUT/EDA_BOX_SalePrice~"+g_i+".png", width = 7, height = 3)
    

#%% 변수 선택(feature selection)

#숫자형 독립변수 Lasso로 추출  
train_x = all[-all["SalePrice"].isnull()][num_var_names]
train_y = all[-all["SalePrice"].isnull()]["SalePrice"]

lasso = LassoCV(normalize=True, random_state = 2019)    
lasso.fit(train_x, train_y)
lasso_coef = lasso.coef_
imp_num_var_names = pd.DataFrame(num_var_names)[lasso_coef!=0][0].tolist()

#범주형 독립변수 Random Forest로 추출
train_x = all[-all["SalePrice"].isnull()][cat_var_names]
#범주형 독립변수 LabelEncoder 변형
for ind in cat_var_names:
    a = LabelEncoder()
    a.fit(train_x[ind])
    train_x[ind] = a.transform(train_x[ind])

train_y = all[-all["SalePrice"].isnull()]["SalePrice"]


rf = RandomForestClassifier(n_estimators=500, random_state = 2019)
rf.fit(train_x,train_y)
imp_df = pd.DataFrame(rf.feature_importances_, cat_var_names, columns = ["score"])

#지니계수가 낮을수록 좋음
rf_imp_df = imp_df.sort_values(by="score", ascending=False)
imp_cat_var_names = rf_imp_df[0:14].index.tolist()

#최종 선택된 중요 변수
imp_var_names = imp_num_var_names + imp_cat_var_names

#%% 예측 분석

###Lasso
#학습데이터
train_x = all[-all["SalePrice"].isnull()][imp_var_names]
train_x = pd.get_dummies(train_x, drop_first = True) #dummy matrix로 변환
train_y - all[-all["SalePrice"].isnull()]["SalePrice"]

#테스트데이터
test_x = all[all["SalePrice"].isnull()][imp_var_names]
test_x = pd.get_dummies(test_x, drop_first = True) #dummy matrix로 변환

#Lasso 적합
lasso1 = LassoCV(normalize=True, random_state = 2019)
lasso1.fit(train_x,train_y)
lasso_pred = lasso1.predict(test_x)
lasso_pred_df = pd.DataFrame(lasso1.predict(train_x),columns = ["predict"])

#train_y 와 lasso_pred_df의 인덱스가 달라서 인덱스를 다시 설정
lasso_pred_df = lasso_pred_df.reset_index(drop=True)
train_y = train_y.reset_index(drop=True)

#train_y와 predict_y 합침
train_y_pred_df = pd.concat([train_y,lasso_pred_df],axis=1)

#산점도 그려보기
ggplot(train_y_pred_df, aes(x="SalePrice", y="predict")) + geom_point(colour = "blue")

#최종 예측값 제출하기
d = {"Id" : test_labels.values, "SalePrice" : lasso_pred}
pd.DataFrame(d).to_csv("./OUT/lasso.submit.csv", index=False, mode = "w") #mode : w(덮어쓰기), a(추가하기)
