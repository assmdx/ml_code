import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
import copy
userProfile_train = pd.read_csv('../data/userProfile_train.csv')
action_train = pd.read_csv('../data/action_train.csv')
orderHistory_train = pd.read_csv('../data/orderHistory_train.csv')
orderFuture_train = pd.read_csv('../data/orderFuture_train.csv')
userComment_train = pd.read_csv('../data/userComment_train.csv')

userProfile_test = pd.read_csv('../data/userProfile_test.csv')
action_test = pd.read_csv('../data/action_test.csv')
orderHistory_test = pd.read_csv('../data/orderHistory_test.csv')
orderFuture_test = pd.read_csv('../data/orderFuture_test.csv')
userComment_test = pd.read_csv('../data/userComment_test.csv')
print orderFuture_train.shape 
print orderFuture_test.shape
le = preprocessing.LabelBinarizer()
encoder1 = le.fit(pd.concat([orderHistory_train,orderHistory_test]).country.values)

le = preprocessing.LabelBinarizer()
encoder2 = le.fit(pd.concat([orderHistory_train,orderHistory_test]).continent.values)
def getHistoryFeature(data, encoder1, encoder2):
    df = copy.deepcopy(data)
    #--get order feature
    feature = df.groupby('userid')['orderType'].agg(['sum','count']).reset_index().rename(columns={'sum': 'order_num_1', 'count': 'order_num'})
    feature['order_num_0'] = feature['order_num'] - feature['order_num_1']
    feature['order_ratio_0'] = feature['order_num_0'].astype('float')/feature['order_num']
    feature['order_ratio_1'] = feature['order_num_1'].astype('float')/feature['order_num']
    f = copy.deepcopy(feature)
    
    #--get total feature
    feature = df.groupby('userid')['city','country','continent'].count().reset_index().rename(
        columns={'city': 'city_num', 'country': 'country_num','continent':'continent_num'})
    f = pd.merge(f,feature, on = 'userid', how = 'left')
    feature = df[df.orderType == 1].groupby('userid')['city','country','continent'].count().reset_index().rename(
        columns={'city': 'city_num_1', 'country': 'country_num_1','continent':'continent_num_1'})
    f = pd.merge(f,feature, on = 'userid', how = 'left').fillna(0)
    for val in ['city_num', 'country_num', 'continent_num']:
        f[val.split('_')[0]+'_ratio_1'] = f[val+'_1'].astype('float')/f[val]
    #--get country feature
#     le = preprocessing.LabelBinarizer()
    country_encoder = encoder1.transform(df.country.values)
    country_encoder_col = ['country_%d'%i for i in range(country_encoder.shape[1])]
    df1 = pd.DataFrame(country_encoder, columns = country_encoder_col)
    df1['userid'] = df['userid'].values
    feature = df1.groupby('userid')[country_encoder_col].agg(['sum','count']).reset_index()
    f = pd.merge(f,feature, on = 'userid', how = 'left')
    
    #--get continent feature
#     le = preprocessing.LabelBinarizer()
    continent_encoder = encoder2.transform(df.continent.values)
    continent_encoder_col = ['continent_%d'%i for i in range(continent_encoder.shape[1])]
    df1 = pd.DataFrame(continent_encoder, columns = continent_encoder_col)
    df1['userid'] = df['userid'].values
    feature = df1.groupby('userid')[continent_encoder_col].agg(['sum','count']).reset_index()
    f = pd.merge(f,feature, on = 'userid', how = 'left')
    
    
    #--get orderTime last feature
#     df1 = df.groupby('userid').apply(lambda x: x.sort_values('orderTime', ascending = False).head(1)).reset_index(drop = True)[['userid','orderid','orderTime','orderType']]
#     df1.columns = [['userid','last_orderid','last_orderTime','last_orderType']]
#     f = pd.merge(f, df1, on = 'userid',how = 'left')
    
    #--get orderTime last 5 feature
#     df1 = df.groupby('userid').apply(lambda x: x.sort_values('orderTime', ascending = False).head(5)).reset_index(drop = True)[['userid','orderid','orderTime','orderType']]
#     df1.columns = [['userid','last_orderid','last_orderTime','last_orderType']]
#     temp = pd.concat([df1,df1.groupby('userid').rank(method = 'first').astype('int').reset_index().rename(
#             columns={'last_orderTime': 'last_orderTime_rank'})['last_orderTime_rank']],axis = 1)
    
#     ff1 = temp.pivot('userid','last_orderTime_rank','last_orderType')
#     ff1.columns = ['his_type%d'%i for i in range(ff1.shape[1])]
#     ff1 = ff1.reset_index()
#     f = pd.merge(f, ff1, on = 'userid',how = 'left')
    
#     ff2 = temp.pivot('userid','last_orderTime_rank','last_orderTime')
#     ff2.columns = ['his_time%d'%i for i in range(ff2.shape[1])]
#     ff2 = ff2.reset_index()
#     f = pd.merge(f, ff2, on = 'userid',how = 'left')
    
    return f
def getUserProfileFeature(df):
    le = preprocessing.LabelBinarizer()
    encoder = le.fit_transform(df.gender.fillna('_NA_').values)
    encoder_col = ['gender_%d'%i for i in range(encoder.shape[1])]
    df1 = pd.DataFrame(encoder, columns = encoder_col)
    df1['userid'] = df['userid'].values
    f = df1
    
    le = preprocessing.LabelBinarizer()
    encoder = le.fit_transform(df.province.fillna('_NA_').values)
    encoder_col = ['province_%d'%i for i in range(encoder.shape[1])]
    df1 = pd.DataFrame(encoder, columns = encoder_col)
    df1['userid'] = df['userid'].values
    f = pd.merge(f,df1, on = 'userid', how = 'left')
    
    le = preprocessing.LabelBinarizer()
    encoder = le.fit_transform(df.age.fillna('_NA_').values)
    encoder_col = ['age_%d'%i for i in range(encoder.shape[1])]
    df1 = pd.DataFrame(encoder, columns = encoder_col)
    df1['userid'] = df['userid'].values
    f = pd.merge(f,df1, on = 'userid', how = 'left')
#     print f.head()
    return f
def getActionFeature(data):
    df = copy.deepcopy(data)
    #---all action feature
    result = df.groupby(['userid','actionType'])['actionTime'].count().reset_index().rename(
        columns = {'actionTime':'actionNum'}).pivot('userid','actionType','actionNum').apply(lambda x: x/np.sum(x))
    result.columns = ['action_type_num%d'%i for i in range(result.shape[1])]
    result = result.reset_index()
    
    #---get feature by unix time
    for window in [6]:
        print window
        print 'actiontype'
        f = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending = False).head(window)).reset_index(drop = True).rename(
            columns={'actionType': 'actionType_last', 'actionTime': 'actionTime_last'})
        f2 = pd.concat([f,f.groupby('userid').rank(method = 'first').astype('int').reset_index().rename(
                columns={'actionTime_last': 'actionTime_last_rank'})['actionTime_last_rank']],axis = 1)
        
        ff1 = f2.pivot('userid','actionTime_last_rank','actionType_last')
        ff1_diff = ff1.diff(1, axis = 1)
        ff1_diff.columns = ['last%d_type_1diff%d'%(window,i) for i in range(ff1_diff.shape[1])]
        ff1_diff = ff1_diff.iloc[:,1:].reset_index()
        result = pd.merge(result, ff1_diff, on = 'userid', how = 'left')
        
        ff1['last%d_type_max'%window] = ff1.max(axis = 0)
        ff1['last%d_type_min'%window] = ff1.min(axis = 0)
        ff1['last%d_type_meam'%window] = ff1.mean(axis = 0)
        ff1['last%d_type_median'%window] = ff1.median(axis = 0)
        ff1['last%d_type_std'%window] = ff1.std(axis = 0)
        ff1['last%d_type_sum'%window] = ff1.sum(axis = 0)
        
        ff1.columns = ['last%d_type%d'%(window,i) for i in range(ff1.shape[1])]
        ff1 = ff1.reset_index()
        
        result = pd.merge(result, ff1, on = 'userid', how = 'left')
        

        print 'actiontime'
        ff2 = f2.pivot('userid','actionTime_last_rank','actionTime_last')
        ff2_diff = ff2.diff(1, axis = 1)
        ff2_diff.columns = ['last%d_time_1diff%d'%(window,i) for i in range(ff2_diff.shape[1])]
        ff2_diff = ff2_diff.iloc[:,1:].reset_index()
        result = pd.merge(result, ff2_diff, on = 'userid', how = 'left')
        
        

        ff2['last%d_time_max'%window] = ff2.max(axis = 0)
        ff2['last%d_time_min'%window] = ff2.min(axis = 0)
        ff2['last%d_time_meam'%window] = ff2.mean(axis = 0)
        ff2['last%d_time_median'%window] = ff2.median(axis = 0)
        ff2['last%d_time_std'%window] = ff2.std(axis = 0)
        ff2['last%d_time_sum'%window] = ff2.sum(axis = 0)
        
        ff2.columns = ['last%d_time%d'%(window,i) for i in range(ff2.shape[1])]
        ff2 = ff2.reset_index()
        result = pd.merge(result, ff2, on = 'userid', how = 'left')
       
        
        ff = f.groupby(['userid','actionType_last'])['actionTime_last'].count().reset_index().rename(
            columns = {'actionTime_last':'actionNum_last'}).pivot('userid','actionType_last','actionNum_last').apply(lambda x: x/np.sum(x))
        ff.columns = ['last%d_action_type_num%d'%(window,i) for i in range(ff.shape[1])]
        ff = ff.reset_index()
        result = pd.merge(result, ff, on = 'userid', how = 'left')
        
    #---sort every type last 1 action feature
    f = df.groupby(['userid','actionType']).apply(lambda x: x.sort_values('actionTime', ascending = False).head(1)).reset_index(drop = True).rename(
        columns={'actionTime': 'type_actionTime_last'})
    ff3 = f.pivot('userid','actionType','type_actionTime_last')
    
    ff3_diff = ff3.diff(1, axis = 1)
    ff3_diff.columns = ['type_%d_lsttime_diff'%i for i in range(ff3_diff.shape[1])]
    ff3_diff = ff3_diff.iloc[:,1:].reset_index()
    ff3.columns = ['type_%d_lasttime'%i for i in range(ff3.shape[1])]
    ff3 = ff3.reset_index()
    result = pd.merge(result, ff3, on = 'userid', how = 'left')
    
    for t in [1,2,3,4,5,6,7,8,9]:
        print t
        window = 5
        df_type = df[df.actionType == t]
        f = df_type.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending = False).head(window)).reset_index(drop = True).rename(
            columns={'actionTime': 'actionTime_last'})
        f2 = pd.concat([f,f.groupby('userid').rank(method = 'first').astype('int').reset_index().rename(
                columns={'actionTime_last': 'actionTime_last_rank'})['actionTime_last_rank']],axis = 1)
        
        ff1 = f2.pivot('userid','actionTime_last_rank','actionTime_last')
        ff1_diff = ff1.diff(1, axis = 1)
        ff1_diff.columns = ['last%d_type%d_time_diff%d'%(window,t,i) for i in range(ff1_diff.shape[1])]
        ff1_diff = ff1_diff.iloc[:,1:].reset_index()
        
        ff1['last%d_type%d_max'%(window,t)] = ff1.max(axis = 0)
        ff1['last%d_type%d_min'%(window,t)] = ff1.min(axis = 0)
        ff1['last%d_type%d_meam'%(window,t)] = ff1.mean(axis = 0)
        ff1['last%d_type%d_median'%(window,t)] = ff1.median(axis = 0)
        ff1['last%d_type%d_std'%(window,t)] = ff1.std(axis = 0)
        ff1['last%d_type%d_sum'%(window,t)] = ff1.sum(axis = 0)
        
        ff1.columns = ['last%d_type%d_time%d'%(window,t,i) for i in range(ff1.shape[1])]
        ff1 = ff1.reset_index()
        
        result = pd.merge(result, ff1, on = 'userid', how = 'left')
        result = pd.merge(result, ff1_diff, on = 'userid', how = 'left')
    #--get feature by date
#     print 'date feature'
#     df['date'] = pd.to_datetime(df['actionTime'],unit='s').dt.date
#     for window in [40]:
#         print window
#         df_select = df[df.date >= pd.bdate_range(end=df.date.max(), periods=window).date[0]]
#         print df_select.shape
#         f = df_select.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending = False).head(6)).reset_index(drop = True).rename(
#                 columns={'actionType': 'actionType_last', 'actionTime': 'actionTime_last'})
#         f2 = pd.concat([f,f.groupby('userid').rank(method = 'first').astype('int').reset_index().rename(
#                 columns={'actionTime_last': 'actionTime_last_rank'})['actionTime_last_rank']],axis = 1)
        
#         ff1 = f2.pivot('userid','actionTime_last_rank','actionType_last')
#         ff1_diff = ff1.diff(1, axis = 1)
#         ff1_diff.columns = ['lastdate%d_type_diff%d'%(window,i) for i in range(ff1_diff.shape[1])]
#         ff1_diff = ff1_diff.iloc[:,1:].reset_index()
        
#         ff1['lastdate%d_type_max'%window] = ff1.max(axis = 0)
#         ff1['lastdate%d_type_min'%window] = ff1.min(axis = 0)
#         ff1['lastdate%d_type_meam'%window] = ff1.mean(axis = 0)
#         ff1.columns = ['lastdate%d_type%d'%(window,i) for i in range(ff1.shape[1])]
#         ff1 = ff1.reset_index()
        
#         result = pd.merge(result, ff1, on = 'userid', how = 'left')
#         result = pd.merge(result, ff1_diff, on = 'userid', how = 'left')
        
#         ff2 = f2.pivot('userid','actionTime_last_rank','actionTime_last')
#         ff2_diff = ff2.diff(1, axis = 1)
#         ff2_diff.columns = ['last%d_time_diff%d'%(window,i) for i in range(ff2_diff.shape[1])]
#         ff2_diff = ff2_diff.iloc[:,1:].reset_index()

#         ff2['last%d_time_max'%window] = ff2.max(axis = 0)
#         ff2['last%d_time_min'%window] = ff2.min(axis = 0)
#         ff2['last%d_time_meam'%window] = ff2.mean(axis = 0)
#         ff2.columns = ['last%d_time%d'%(window,i) for i in range(ff2.shape[1])]
#         ff2 = ff2.reset_index()
#         result = pd.merge(result, ff2, on = 'userid', how = 'left')
#         result = pd.merge(result, ff2_diff, on = 'userid', how = 'left')
        
        
#         ff = f.groupby(['userid','actionType_last'])['actionTime_last'].count().reset_index().rename(
#             columns = {'actionTime_last':'actionNum_last'}).pivot('userid','actionType_last','actionNum_last').apply(lambda x: x/np.sum(x))
#         ff.columns = ['last%d_action_type_num%d'%(window,i) for i in range(ff.shape[1])]
#         ff = ff.reset_index()
#         result = pd.merge(result, ff, on = 'userid', how = 'left')

    
    
    return result
from sklearn.preprocessing import MultiLabelBinarizer
def getCommentFeature(data):
    df = copy.deepcopy(data)
    feature = df.groupby('userid')['rating'].agg(['max','min','mean','sum','count','median','std']).reset_index().rename(
        columns={'max': 'rate_max', 'min': 'rate_min','mean':'rate_mean','sum':'rate_sum','count':'rate_count','median':'rate_median','std':'rate_std'})
    
#     mlb = MultiLabelBinarizer()
#     tagsV = []
#     for line in df.tags.values:
#         if line == line:
# #             print line
#             tagsV.append(set(line.split('|')))
#         else:
#             tagsV.append(set(''))
            
    
#     tagsF = mlb.fit_transform(tagsV)
#     name = []
#     for i in range(tagsF.shape[1]):
#         df['tag_%d'%i] = tagsF[:,i]
#         name.append('tag_%d'%i)
    
#     f = df.groupby('userid')[name].agg(['mean','sum','count','std']).reset_index()
    
#     feature = pd.merge(feature, f, on= 'userid', how = 'left') 
    
    
    return feature

def feature_important(bst):
    feature_score = bst.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    x1 = []
    x2 = []
    for (key,value) in feature_score:
        x1.append(key)
        x2.append(value)
    feat_im = pd.DataFrame({"feature_name":x1, "score":x2})
    feat_im.to_csv("../model/feature_important.20170911.csv", index=False)

history = getHistoryFeature(pd.concat([orderHistory_train, orderHistory_test]), encoder1, encoder2)
train = pd.merge(orderFuture_train, history, on = 'userid', how = 'left')
test = pd.merge(orderFuture_test, history, on = 'userid', how = 'left')
print train.shape, test.shape

profile = getUserProfileFeature(pd.concat([userProfile_train, userProfile_test]))
train = pd.merge(train, profile, on = 'userid', how = 'left')
test = pd.merge(test, profile, on = 'userid', how = 'left')
print train.shape, test.shape

comment = getCommentFeature(pd.concat([userComment_train, userComment_test]))
train = pd.merge(train, comment, on = 'userid', how = 'left')
test = pd.merge(test, comment, on = 'userid', how = 'left')
print train.shape, test.shape

action = getActionFeature(pd.concat([action_train, action_test]))
train = pd.merge(train, action, on = 'userid', how = 'left')
test = pd.merge(test, action, on = 'userid', how = 'left')
print train.shape, test.shape
train_x = train.drop(['orderType'], axis = 1)
train_y = train.orderType.values
print train_x.shape, len(train_y)

param = {}
param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['stratified'] = 'True'
param['eta'] = 0.02
param['silent'] = 1
param['max_depth'] = 5
param['subsample'] = 0.7
param['colsample_bytree'] = 0.8
# param['lambda'] = 2
# param['min_child_weight'] = 10
param['scale_pos_weight'] = 1
param['seed'] = 1024
param['nthread'] = 16

dtrain = xgb.DMatrix(train_x, label=train_y)
# res = xgb.cv(param, dtrain, 3500, nfold = 5, early_stopping_rounds=100, verbose_eval = 20)
train_x = train.drop(['orderType'], axis = 1)
train_y = train.orderType.values
print train_x.shape, len(train_y)

param = {}
param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['stratified'] = 'True'
param['eta'] = 0.02
param['silent'] = 1
param['max_depth'] = 5
param['subsample'] = 0.7
param['colsample_bytree'] = 0.8
# param['lambda'] = 2
# param['min_child_weight'] = 10
param['scale_pos_weight'] = 1
param['seed'] = 1024
param['nthread'] = 16

dtrain = xgb.DMatrix(train_x, label=train_y)
# res = xgb.cv(param, dtrain, 3500, nfold = 5, early_stopping_rounds=100, verbose_eval = 20)
print res.shape[0]
model = xgb.train(param, dtrain, res.shape[0], evals=[(dtrain,'train')], verbose_eval=500)
feature_important(model)#??????????????????
test_x = test[train_x.columns.values]
dtest = xgb.DMatrix(test_x)
y = model.predict(dtest)
test['orderType']=y
test[['userid','orderType']].to_csv('../result/xgb_baseline_local9594.csv',index=False)
imp_f = model.get_fscore()
imp_df = pd.DataFrame({'feature':[key for key, value in imp_f.items()], 'fscore':[value for key, value in imp_f.items()]})
imp_df = imp_df.sort_values(by = 'fscore',ascending=False)
imp_df.to_csv('../imp/xgb_imp.csv', index = False)
train_x = train.drop(['orderType'], axis = 1)
train_y = train.orderType.values
print train_x.shape, len(train_y)

import lightgbm as lgb

param = {}
param['task'] = 'train'
param['boosting_type'] = 'gbdt'
param['objective'] = 'binary'
param['metric'] = 'auc'
param['min_sum_hessian_in_leaf'] = 0.1
param['learning_rate'] = 0.01
param['verbosity'] = 2
param['tree_learner'] = 'feature'
param['num_leaves'] = 128
param['feature_fraction'] = 0.7
param['bagging_fraction'] = 0.7
param['bagging_freq'] = 1
param['num_threads'] = 16


dtrain = lgb.Dataset(train_x, label=train_y)
res1gb = lgb.cv(param, dtrain, 5500, nfold = 5, early_stopping_rounds=100, verbose_eval = 20)
ro = len(res1gb['auc-mean'])
model = lgb.train(param, dtrain, ro, valid_sets=[dtrain], verbose_eval=500)
test_x = test[train_x.columns.values]
y = model.predict(test_x)
test['orderType']=y
test[['userid','orderType']].to_csv('../result/lgb_baseline_local9619.csv',index=False)