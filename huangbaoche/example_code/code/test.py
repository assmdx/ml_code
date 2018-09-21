# # -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

def fea8(df,tt,userComment):
	# 获取评分的最小值，最大值，方差
	tdf = userComment.groupby('userid')['rating'].min().reset_index().rename(columns={'rating':'min_rating'})
	df = pd.merge(df,tdf,on='userid',how='left')
	tdf = userComment.groupby('userid')['rating'].max().reset_index().rename(columns={'rating':'max_rating'})
	df = pd.merge(df,tdf,on='userid',how='left')
	tdf = userComment.groupby('userid')['rating'].var().reset_index().rename(columns={'rating':'max_rating'})
	df = pd.merge(df,tdf,on='userid',how='left')
	return df

def fea10(df,tt,userProfile):
	#提取省份,性别，年龄 one-hot特征	
	one_hot = pd.get_dummies(userProfile['province'])	
	userProfile = userProfile.join(one_hot)
	a = 0
	for typeno in set(userProfile.province):
		userProfile = userProfile.rename(columns = {typeno:'province_'+ str(a)})
		a+=1
	userProfile = userProfile.drop(['province'],axis =1)
	one_hot2 = pd.get_dummies(userProfile['gender'])
	userProfile = userProfile.join(one_hot2)
	a = 0
	for typeno in set(userProfile.gender):
		userProfile = userProfile.rename(columns = {typeno:'gender_'+ str(a)})
		a+=1
	userProfile = userProfile.drop(['gender'],axis =1)
	one_hot3 = pd.get_dummies(userProfile['age'])
	userProfile = userProfile.join(one_hot3)
	a = 0
	for typeno in set(userProfile.age):
		userProfile = userProfile.rename(columns = {typeno:'age_'+ str(a)})
		a+=1
	userProfile = userProfile.drop(['age'],axis =1)
	# userProfile.to_csv("feature/fea10_"+tt+".csv", index=False, encoding="utf-8")
	return userProfile

def fea11(action):           
    df = action[['userid']]  
    for typeno in set(action.actionType):       
        tdf = action[action.actionType==typeno].groupby('userid').size().reset_index().rename(columns={0:'actionType_i'+str(typeno)})
        if typeno == 1:
            df = pd.merge(df,tdf, how="left", on="userid") 
            continue
        if typeno < 5 :            
            df['actionType_i1_i4'] = df['actionType_i1'] + tdf['actionType_i' + str(typeno)]
            continue
        if typeno == 5:
            df = pd.merge(df,tdf, how="left", on="userid") 
            continue
        if typeno > 5 :            
            df['actionType_i5_i9'] = df['actionType_i5'] + tdf['actionType_i' + str(typeno)]
        if typeno == 9:
        	df['actionType_i9_div_i1_to_i4'] = tdf['actionType_i' + str(typeno)]/(df['actionType_i1_i4'] + 1) 
    df = df.drop(['actionType_i1'],axis=1)
    df = df.drop(['actionType_i5'],axis=1)
    df.to_csv('samdgkajs.csv', index=False, encoding="utf-8")
    return df

def fea12( df, tt ,action):    
    curstamp = action['actionTime'].max()
    daysec = 86400  
    days = [7,180,360]

    #最近7,180,360天内actionType 1~9次数
    for typeno in set(action.actionType):      
        for day in days:
            tdf = action[(action.actionType==typeno) & (abs(action.actionTime-curstamp)<=day*daysec)].groupby('userid').size().\
                    reset_index().rename(columns={0:'actype_'+str(typeno)+'_num_day_'+str(day)})     
            df = pd.merge(df, tdf, on='userid', how='left')            
    # df.to_csv("feature/fea12_"+tt+".csv", index=False, encoding="utf-8")
    return df
def fea9(userComment):	
	df = userComment[['userid']]
	userComment['tags'] = userComment['tags'].apply(func_fea9)
	userComment['commentsKeyWords'] = userComment['commentsKeyWords'].apply(func_fea9)
	userComment['tags'] = userComment['tags']/4 +  userComment['commentsKeyWords']
	userComment['rating'] = userComment['rating'].apply(lambda x:(x-2.5))
	userComment['tags'] = userComment['tags']*userComment['rating']
	tdf = userComment['tags'].groupby(userComment['userid']).mean().reset_index().rename(columns={'tags':'comment_value'})
	df = pd.merge(df, tdf, on='userid', how='left')
	return df

def func_fea9(x):
	if str(x) == 'nan':
		return 0
	else:
		return len(x)	
def fea13(action):
	df = action[['userid']]  
	for typeno in set(action.actionType):
		tdf = action[action.actionType==typeno].groupby('userid').size().reset_index().rename(columns={0:'actionType_i'+str(typeno)})
		if typeno == 1:
			df = pd.merge(df,tdf, how="left", on="userid") 
			continue
		if typeno < 5 :            
			df['actionType_i1_i4'] = df['actionType_i1'] + tdf['actionType_i' + str(typeno)]
			continue
		if typeno == 5:
			df = pd.merge(df,tdf, how="left", on="userid") 
			continue
		if typeno > 5 :            
			df['actionType_i5_i9'] = df['actionType_i5'] + tdf['actionType_i' + str(typeno)]
	df['actionType_i5_i9_div_i1_i4'] = df['actionType_i5_i9'] / (df['actionType_i1_i4'].apply(lambda x:x+1)) 	
	df = df.drop(['actionType_i1'],axis=1)	
	df = df.drop(['actionType_i5'],axis=1)	
	df = df.drop(['actionType_i1_i4'],axis=1)	
	df = df.drop(['actionType_i5_i9'],axis=1)	
	df.to_csv('samdgkajs.csv', index=False, encoding="utf-8")
	return df

def fea14(action,orderHistory):
	#ordertype的次数/action 的总数
	df = orderHistory[['userid']]
	edf = orderHistory['orderType'].groupby(orderHistory['userid']).sum().reset_index().rename(columns={0:'orderType'})	
	tdf = action['actionType'].groupby(action['userid']).size().reset_index().rename(columns = {0:'actionType'})	
	df['orderhistory_div_allactionType'] = edf['orderType']/(tdf['actionType']+1)
	for i in range(5,9,1):		
		fdf = action[action.actionType == i].groupby(action['userid']).size().reset_index().rename(columns = {0:'actionType'})
		df['orderhistory_div_allactionType'+str(i)] = edf['orderType']/(fdf['actionType']+1)		
	df.to_csv('okkkokoko.csv', index=False, encoding="utf-8")
	return df

userProfile_train = pd.read_csv('../data/trainingset/userProfile_train.csv')
action_train = pd.read_csv('../data/trainingset/action_train.csv')
orderHistory_train = pd.read_csv('../data/trainingset/orderHistory_train.csv')
orderFuture_train = pd.read_csv('../data/trainingset/orderFuture_train.csv')
userComment_train = pd.read_csv('../data/trainingset/userComment_train.csv')

userProfile_test = pd.read_csv('../data/test/userProfile_test.csv')
action_test = pd.read_csv('../data/test/action_test.csv')
orderHistory_test = pd.read_csv('../data/test/orderHistory_test.csv')
orderFuture_test = pd.read_csv('../data/test/orderFuture_test.csv')
userComment_test = pd.read_csv('../data/test/userComment_test.csv')

# userComment_train=pd.read_csv("data/userComment_train.csv",dtype={'tags':str},encoding="utf-8")
# orderFuture = pd.read_csv("data/orderFuture_train.csv", encoding="utf-8")
# userProfile_train = pd.read_csv("data/userProfile_train.csv", encoding="utf-8")
# action_train = pd.read_csv("data/action_train.csv", encoding="utf-8")
# df = orderFuture[['userid']]
# fea8(df,'train',userComment_train)
# def fea10(df,tt,userComment):
# userProfile_train = pd.read_csv("../data/userProfile_train.csv", encoding="utf-8")
df = orderFuture_train[['userid']]

fea_14 = fea14(pd.concat([action_train, action_test]),pd.concat([orderHistory_train, orderHistory_test]))
df = pd.merge(df, fea_14, on='userid', how='left')
# fea12(df,'train',action_train)

# x = 2
# print(int (x > 3))
# fea9(df,userComment_train)