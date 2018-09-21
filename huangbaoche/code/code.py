# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time


orderFuture_train = pd.read_csv("../data/orderFuture_train.csv", encoding="utf-8")
userProfile_train = pd.read_csv("../data/userProfile_train.csv", encoding="utf-8")
action_train = pd.read_csv("../data/action_train.csv", encoding="utf-8")
orderHistory_train = pd.read_csv("../data/orderHistory_train.csv", encoding="utf-8")
userComment_train = pd.read_csv("../data/userComment_train.csv", encoding="utf-8")

orderFuture_test = pd.read_csv("../data/orderFuture_test.csv", encoding="utf-8")
userProfile_test = pd.read_csv("../data/userProfile_test.csv", encoding="utf-8")
action_test = pd.read_csv("../data/action_test.csv", encoding="utf-8")
orderHistory_test = pd.read_csv("../data/orderHistory_test.csv", encoding="utf-8")
userComment_test = pd.read_csv("../data/userComment_test.csv", encoding="utf-8")


#orderFuture_train = pd.read_csv("../data/orderFuture_train.csv", encoding="utf-8", nrows=10000)
#userProfile_train = pd.read_csv("../data/userProfile_train.csv", encoding="utf-8", nrows=10000)
#action_train = pd.read_csv("../data/action_train.csv", encoding="utf-8", nrows=10000)
#orderHistory_train = pd.read_csv("../data/orderHistory_train.csv", encoding="utf-8", nrows=10000)
#userComment_train = pd.read_csv("../data/userComment_train.csv", encoding="utf-8", nrows=10000)
#
#orderFuture_test = pd.read_csv("../data/orderFuture_test.csv", encoding="utf-8", nrows=10000)
#userProfile_test = pd.read_csv("../data/userProfile_test.csv", encoding="utf-8", nrows=10000)
#action_test = pd.read_csv("../data/action_test.csv", encoding="utf-8", nrows=10000)
#orderHistory_test = pd.read_csv("../data/orderHistory_test.csv", encoding="utf-8", nrows=10000)
#userComment_test = pd.read_csv("../data/userComment_test.csv", encoding="utf-8", nrows=10000)


def XGB():
    num_round = 5000
    cv = 1 #本地cv时为1，线上预测时为0
    
    train = pd.read_csv("../feature/fea_train.csv", encoding="utf-8")
    test = pd.read_csv("../feature/fea_test.csv", encoding="utf-8")
    train_y = train['label']
    train_x = train.drop(['userid', 'label'], axis=1)
    ids = test['userid']
    test = test.drop(['userid', 'label'], axis=1)
    
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test)
    
    params={
        'booster':'gbtree', #gbtree   dart gblinear
        'objective': 'binary:logistic',
#        'scale_pos_weight': float(np.sum(train_y == 0)) / np.sum(train_y==1),      
        'eval_metric': 'auc',
    
        'max_depth':12, #8
        'min_child_weight':5, #
        
        'subsample':0.7, #0.7
        'colsample_bytree':0.7, #0.6
        
        'lambda':10,   #3
    #    'alpha': 0.5823428683068832,
    #    'lambda_bias': 0.01,
        'gamma':4, #4
        
        'eta': 0.02,  #0.015
        'seed':27,
        'nthread':16,
        #'tree_method':'gpu_hist',
        
        #'tree_method': 'gpu_hist',
        #'n_gpus':-1,
        #'updater':'grow_gpu',
        #'gpu_id':1,
        
        'stratified':True,
        'silent':1
        } 
        #nu
    
    
    if cv:
        sco = xgb.cv(params, dtrain, num_round, nfold=10, metrics={'auc'}, early_stopping_rounds=100, \
                   verbose_eval = 5, seed = 27)
        
        mean_auc = sco['test-auc-mean']
        last_auc = np.array(mean_auc).max()
        print ('max auc:', last_auc)
        print('feature:', train.columns)
    else:
        evallist  = [(dtrain,'train')]
        model = xgb.train( params, dtrain, num_round, evallist, verbose_eval = 5)
        feature_important(bst)
        ptest = model.predict(dtest)
        subm = pd.DataFrame()
        subm['userid'] = ids
        subm['orderType'] = ptest
        subm.to_csv("../submit/base_subm.csv", encoding="utf-8", index=False)
    

def labelEncoder(train, test, cols):
    #对非数值列cols进行编码
    test['label'] = -1
    test = test[train.columns]
    df = pd.concat([train, test], axis=0)
    for col in cols:
        df[col] = pd.Categorical(df[col].values).labels
    n = train.shape[0]
    train = df.iloc[:n,:]
    test = df.iloc[n:,:]
    return train, test  

def feature_important(bst):
    feature_score = bst.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    x1 = []
    x2 = []
    for (key,value) in feature_score:
        x1.append(key)
        x2.append(value)
    feat_im = pd.DataFrame({"feature_name":x1, "score":x2})
    feat_im.to_csv("model/feature_important.20170911.csv", index=False)
   

def base_fea(tt, df, userProfile, action, orderHistory, userComment):       
    df = pd.merge(df, userProfile, how="left", on="userid")

    #用户行为1~9的总次数
    for typeno in set(action.actionType):
        tdf = action[action.actionType==typeno].groupby('userid').size().reset_index().rename(columns={0:'actionType_'+str(typeno)})
        df = pd.merge(df, tdf, on='userid', how='left')
    
    #用户购买0,1服务的总次数,是否购买过0,1服务
    for typeno in set(orderHistory.orderType):
        tdf = orderHistory[orderHistory.orderType==typeno].groupby('userid').size().reset_index().rename(columns={0:'orderType_'+str(typeno)})
        df = pd.merge(df, tdf, on='userid', how="left")
        name = 'buy_order_'+str(typeno)
        tdf = tdf.rename(columns={'orderType_'+str(typeno):name})
        tdf[name] = tdf[name].apply(lambda x:int(x>0))
        df = pd.merge(df, tdf, on='userid', how="left")
           
    #用户历史评分平均值
    tdf = userComment['rating'].groupby(userComment['userid']).mean().reset_index().rename(columns={'rating':'avg_rating'})
    df = pd.merge(df, tdf, on="userid", how="left")
    
    if tt=="train":
        df = df.rename(columns={'orderType':'label'})
        
    df.to_csv("../feature/fea0_"+tt+".csv", index=False, encoding="utf-8")


def fea1(tt, df, action):    
    curstamp = action['actionTime'].max()
    daysec = 86400  
    days = [1,2,3,5,8,13,30,60,90]

    #最近1，2,3,5,8,13,30,60,90天内actionType 1~9次数
    for typeno in set(action.actionType):      
        for day in days:
            tdf = action[(action.actionType==typeno) & (abs(action.actionTime-curstamp)<=day*daysec)].groupby('userid').size().\
                    reset_index().rename(columns={0:'actype_'+str(typeno)+'_num_day_'+str(day)})     
            df = pd.merge(df, tdf, on='userid', how='left')
            
    df.to_csv("../feature/fea1_"+tt+".csv", index=False, encoding="utf-8")


def fea2(tt, action):            
    #最后一次action的type
    tdf = action['actionTime'].groupby(action['userid']).max().reset_index()

    m = {}
    for r in tdf.values:
        m[r[0]] = r[1]

    lastTypes = []
    for uid in tdf.userid:
        stamp = m[uid]
        ttype = action.loc[action.actionTime==stamp, 'actionType'].values[0]
        lastTypes.append(ttype)
        
    tdf['lastActionType'] = lastTypes
    tdf = tdf.drop(['actionTime'], axis=1)
  
    tdf.to_csv("../feature/fea2_"+tt+".csv", index=False, encoding="utf-8")
    
    
def fea3(tt, action):
    #离最近的3、5、8、9的距离
    tdf = action.groupby('userid').apply(func2).reset_index().rename(columns={0:'dis'}) 
    names = ['dis_of_actype3','dis_of_actype5','dis_of_actype8','dis_of_actype9']
    tdf = splitColumn(tdf, 'dis', names)          
    tdf.to_csv("../feature/fea3_"+tt+".csv", index=False, encoding="utf-8")

def func2(df):
    tdf = df.groupby('actionType')['actionTime'].max().reset_index()
    df.sort_values(by='actionTime', ascending=True)
    df = df.reset_index(drop=True)
    re = []
       
    types = [3,5,8,9]
    size = df.shape[0]

    for ttype in types:
        if len(tdf.loc[tdf.actionType==ttype, 'actionTime']):
            stamp = tdf.loc[tdf.actionType==ttype, 'actionTime'].values[0]
            pos = df.loc[(df.actionType==ttype)&(df.actionTime==stamp)].index[0]
            dis = size-pos
            re.append(dis)
        else:
            re.append(np.nan)
    
    return re  


def fea4(tt, df, orderHistory):
    curstamp = orderHistory['orderTime'].max()
    daysec = 86400  
    days = [1,2,3,5,8,13,30,60,90]

    #最近1，2,3,5,8,13,30,60,90天内orderType 0、1次数
    for typeno in set(orderHistory.orderType):
        for day in days:
            tdf = orderHistory[(orderHistory.orderType==typeno) & (abs(orderHistory.orderTime-curstamp)<=day*daysec)] \
                    .groupby('userid').size().reset_index().rename(columns={0:'ordertype_'+str(typeno)+'_num_day_'+str(day)})
            df = pd.merge(df, tdf, on='userid', how='left')
            
    df.to_csv("../feature/fea4_"+tt+".csv", index=False, encoding="utf-8")


def fea5(tt, orderHistory):      
    #最后一次order的type
    tdf = orderHistory['orderTime'].groupby(orderHistory['userid']).max().reset_index()
    m = {}
    for r in tdf.values:
        m[r[0]] = r[1]

    lastTypes = []
    for uid in tdf.userid:
        stamp = m[uid]
        ttype = orderHistory.loc[orderHistory.orderTime==stamp, 'orderType'].values[0]
        lastTypes.append(ttype)
        
    tdf['lastOrderType'] = lastTypes
    tdf = tdf.drop(['orderTime'], axis=1)    
    tdf.to_csv("../feature/fea5_"+tt+".csv", index=False, encoding="utf-8")
   
    
def fea6(tt, orderHistory):
    #orderTime的时间间隔的最小值、均值、方差、最后四个时间间隔值、最后四个时间间隔均值、方差
    tdf = orderHistory.groupby(orderHistory['userid'])['orderTime'].apply(func1).reset_index()
    names = ['min_time_interval','mean_time_interval','var_time_interval','last_interval_1', \
            'last_interval_2','last_interval_3','last_interval_4','mean_last4interval','var_last4interval']
    tdf = splitColumn(tdf, 'orderTime', names)
    tdf.to_csv("../feature/fea6_"+tt+".csv", index=False, encoding="utf-8")  

def func1(x):
    x = x.apply(lambda ix: ix/float(360))
    x = list(x)
    
    if(len(x)<2):
        return [np.nan]*9
    
    x.sort()
    
    y = []
    for i in range(1,len(x),1):
        y.append(x[i]-x[i-1])
        
    y = np.array(y)
    re = [y.min(),y.mean(),y.var()]
    
    y = y[-4:]
    t = [np.nan]*(4-len(y))
    t.extend(list(y))
    
    re.extend(t)
    re.extend([y.mean(), y.var()])
    
    return re


def fea7(tt, df, action):
    #离最近的1、2、3、4、5、6、7、8、9的时间
    curstamp = action['actionTime'].max()

    for typeno in set(action.actionType):
        tdf = action[action.actionType==typeno].groupby('userid')['actionTime'].max().reset_index()
        tdf['actionTime'] = tdf['actionTime'].apply(lambda x:curstamp-x)
        tdf = tdf.rename(columns={'actionTime':'time_of_last_'+str(typeno)})
        df = pd.merge(df, tdf, on='userid', how='left')
    
    df.to_csv("../feature/fea7_"+tt+".csv", index=False, encoding="utf-8")  
    

def splitColumn(df, colname, split_names):
    #df的colname列为数组，将colname列分割，列名在split_names中
    for i in range(len(split_names)):
        df.insert(1, split_names[i], df[colname])
        df[split_names[i]] = df[split_names[i]].map(lambda x:x[i])
    df = df.drop([colname], axis=1)    
    return df 
    

# def fea8(df,tt,userComment):
# 	# 获取评分的最小值，最大值，方差
# 	tdf = userComment.groupby('userid')['rating'].min().reset_index().rename(columns={'rating':'min_rating'})
# 	df = pd.merge(df,tdf,on='userid',how='left')
# 	tdf = userComment.groupby('userid')['rating'].max().reset_index().rename(columns={'rating':'max_rating'})
# 	df = pd.merge(df,tdf,on='userid',how='left')
# 	tdf = userComment.groupby('userid')['rating'].var().reset_index().rename(columns={'rating':'max_rating'})
# 	df = pd.merge(df,tdf,on='userid',how='left')
# 	df.to_csv("../feature/fea8_"+tt+".csv", index=False, encoding="utf-8")

# def fea9(df,tt,userComment):
# 	# 获取评论内容和标签的和, 加权值 均值
# 	tdf = userComment.groupby('userid')['tags'].apply(fun_fea9).reset_index().rename(columns={'tags':'tags_num'})
# 	edf = userComment.groupby('userid')['commentsKeyWords'].apply(fun2_fea9).reset_index().rename(columns={'commentsKeyWords':'com_num'})
# 	df = pd.merge(df,tdf,on='userid',how='left')
# 	df = pd.merge(df,edf,on='userid',how='left')
# 	df['tags'] = df['tags'] +  df['commentsKeyWords']
# 	df.drop(['commentsKeyWords'], axis=1)
# 	df = df.groupby('userid')['tags'].mean().reset_index().rename(columns={'tags':'tags_avg'})	
# 	df.to_csv("../feature/fea9_"+tt+".csv", index=False, encoding="utf-8")
# def fun_fea9(x):
# 	y = []	
# 	for i in range(1,len(x),1):
#         y.append(GetWordsNum(x[i].replace("|","").decode('gbk'))/4)
#     return y    
# def fun2_fea9(x):
# 	y = []	
# 	for i in range(1,len(x),1):		
#         y.append(GetWordsNum(x[i].replace("[","").replace("]","").replace("\"","").decode('gbk')))
#     return y

def conbine_features():
    train = pd.read_csv("../feature/fea0_train.csv", encoding="utf-8")
    test = pd.read_csv("../feature/fea0_test.csv", encoding="utf-8")
    
    colnames = ['gender', 'province', 'age']      
    train, test = labelEncoder(train, test, colnames)
    
    feaNums = range(1, 8, 1) ########################
    for num in feaNums:
        df_train = pd.read_csv("../feature/fea"+str(num)+"_train.csv", encoding="utf-8")
        df_test = pd.read_csv("../feature/fea"+str(num)+"_test.csv", encoding="utf-8")
        train = pd.merge(train, df_train, on='userid', how='left')
        test = pd.merge(test, df_test, on='userid', how='left')
       
    train.to_csv("../feature/fea_train.csv", index=False, encoding="utf-8")
    test.to_csv("../feature/fea_test.csv", index=False, encoding="utf-8")
    
    
def fea(tt, orderFuture, userProfile, action, orderHistory, userComment):
    #分别生成不同的特征，已生成的特征无需再次生成
    df = orderFuture[['userid']]
    
    base_fea(tt, orderFuture, userProfile, action, orderHistory, userComment)
    fea1(tt, df, action)    
    fea2(tt, action)
    fea3(tt, action)
    fea4(tt, df, orderHistory)
    fea5(tt, orderHistory)
    fea6(tt, orderHistory)
    fea7(tt, df, action)    


def gen_features():
    fea('train', orderFuture_train, userProfile_train, action_train, orderHistory_train, userComment_train)
    fea('test', orderFuture_test, userProfile_test, action_test, orderHistory_test, userComment_test)

    
def main():
    genFea = 0
    if genFea:
        gen_features()
    else:
        conbine_features()
        XGB()
        
        
if __name__ == "__main__":
    main()