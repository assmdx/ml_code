import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing

version = 10


# def fea8(userComment):
#     df = userComment[['userid']]
#     tdf = userComment.groupby('userid')['rating'].min().reset_index().rename(columns={'rating':'min_rating'})
#     df = pd.merge(df,tdf,on='userid',how='left')
#     tdf = userComment.groupby('userid')['rating'].max().reset_index().rename(columns={'rating':'max_rating'})
#     df = pd.merge(df,tdf,on='userid',how='left')
#     tdf = userComment.groupby('userid')['rating'].var().reset_index().rename(columns={'rating':'max_rating'})
#     df = pd.merge(df,tdf,on='userid',how='left')
#     return df

# def fea9(userComment):  
#     df = userComment[['userid']]
#     userComment['tags'] = userComment['tags'].apply(func_fea9)
#     userComment['commentsKeyWords'] = userComment['commentsKeyWords'].apply(func_fea9)
#     userComment['tags'] = userComment['tags']/4 +  userComment['commentsKeyWords']
#     userComment['rating'] = userComment['rating'].apply(lambda x:(x-2.5))
#     userComment['tags'] = userComment['tags']*userComment['rating']
#     tdf = userComment['tags'].groupby(userComment['userid']).mean().reset_index().rename(columns={'tags':'comment_value'})
#     df = pd.merge(df, tdf, on='userid', how='left')
#     return df
# def func_fea9(x):
#     if str(x) == 'nan':
#         return 0
#     else:
#         return len(x) 
# def fea10(userProfile):
#     df = userProfile[['userid']]  
#     one_hot = pd.get_dummies(userProfile['province'])   
#     userProfile = userProfile.join(one_hot)
#     a = 0
#     for typeno in set(userProfile.province):
#         userProfile = userProfile.rename(columns = {typeno:'province_'+ str(a)})
#         a+=1
#     userProfile = userProfile.drop(['province'],axis =1)
#     one_hot2 = pd.get_dummies(userProfile['gender'])
#     userProfile = userProfile.join(one_hot2)
#     a = 0
#     for typeno in set(userProfile.gender):
#         userProfile = userProfile.rename(columns = {typeno:'gender_'+ str(a)})
#         a+=1
#     userProfile = userProfile.drop(['gender'],axis =1)
#     one_hot3 = pd.get_dummies(userProfile['age'])
#     userProfile = userProfile.join(one_hot3)
#     a = 0
#     for typeno in set(userProfile.age):
#         userProfile = userProfile.rename(columns = {typeno:'age_'+ str(a)})
#         a+=1
#     userProfile = userProfile.drop(['age'],axis =1)
#     # userProfile.to_csv("feature/fea10_"+tt+".csv", index=False, encoding="utf-8")
#     return userProfile
    

# def fea11(action):           
#     df = action[['userid']]  
#     for typeno in set(action.actionType):       
#         tdf = action[action.actionType==typeno].groupby('userid').size().reset_index().rename(columns={0:'actionType_i'+str(typeno)})
#         if typeno == 1:
#             df = pd.merge(df,tdf, how="left", on="userid") 
#             continue
#         if typeno < 5 :            
#             df['actionType_i1_i4'] = df['actionType_i1'] + tdf['actionType_i' + str(typeno)]
#             continue
#         if typeno == 5:
#             df = pd.merge(df,tdf, how="left", on="userid") 
#             continue
#         if typeno > 5 :            
#             df['actionType_i5_i9'] = df['actionType_i5'] + tdf['actionType_i' + str(typeno)]
#         if typeno == 9:
#             df['actionType_i9_div_i1_to_i4'] = tdf['actionType_i' + str(typeno)]/(df['actionType_i1_i4'] + 1) 
#     df = df.drop(['actionType_i1'],axis=1)
#     df = df.drop(['actionType_i5'],axis=1)    
#     return df

# def fea12(action): 
#     df = action[['userid']]  
#     curstamp = action['actionTime'].max()
#     daysec = 86400  
#     days = [7,180,360]
    
#     for typeno in set(action.actionType):      
#         for day in days:
#             tdf = action[(action.actionType==typeno) & (abs(action.actionTime-curstamp)<=day*daysec)].groupby('userid').size().\
#                     reset_index().rename(columns={0:'actype_'+str(typeno)+'_num_dayday_'+str(day)})     
#             df = pd.merge(df, tdf, on='userid', how='left')                
#     return df
# def fea13(action):
#     df = action[['userid']]  
#     for typeno in set(action.actionType):
#         tdf = action[action.actionType==typeno].groupby('userid').size().reset_index().rename(columns={0:'actionType_i'+str(typeno)})
#         if typeno == 1:
#             df = pd.merge(df,tdf, how="left", on="userid") 
#             continue
#         if typeno < 5 :            
#             df['actionType_i1_i4'] = df['actionType_i1'] + tdf['actionType_i' + str(typeno)]
#             continue
#         if typeno == 5:
#             df = pd.merge(df,tdf, how="left", on="userid") 
#             continue
#         if typeno > 5 :            
#             df['actionType_i5_i9'] = df['actionType_i5'] + tdf['actionType_i' + str(typeno)]
#     df['actionType_i5_i9_div_i1_i4'] = df['actionType_i5_i9'] / (df['actionType_i1_i4'].apply(lambda x:x+1))    
#     df = df.drop(['actionType_i1'],axis=1)  
#     df = df.drop(['actionType_i5'],axis=1)  
#     df = df.drop(['actionType_i1_i4'],axis=1)   
#     df = df.drop(['actionType_i5_i9'],axis=1)       
#     return df

def XGB(train, test):
    train_x = train.drop(['orderType', 'userid'], axis=1)
    train_y = train.orderType.values
    print(train_x.shape)
    print(len(train_y))

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

    res = xgb.cv(param, dtrain, 3500, nfold=5, early_stopping_rounds=100, verbose_eval=20)

    model = xgb.train(param, dtrain, res.shape[0], evals=[(dtrain, 'train')], verbose_eval=500)
    test_x = test[train_x.columns.values]
    dtest = xgb.DMatrix(test_x)
    y = model.predict(dtest)
    test['orderType'] = y
    test[['userid', 'orderType']].to_csv('../result/xgb_baseline' + str(version) + '.csv', index=False)

    imp_f = model.get_fscore()
    imp_df = pd.DataFrame(
        {'feature': [key for key, value in imp_f.items()], 'fscore': [value for key, value in imp_f.items()]})
    imp_df = imp_df.sort_values(by='fscore', ascending=False)
    imp_df.to_csv('../imp/xgb_imp' + str(version) + '.csv', index=False)


def LGB(train,test):
    train_x = train.drop(['orderType','userid'], axis=1)
    train_y = train.orderType.values

    print(train_x.shape)
    print(len(train_y))

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
    res1gb = lgb.cv(param, dtrain, 5500, nfold=5, early_stopping_rounds=100, verbose_eval=20)
    ro = len(res1gb['auc-mean'])
    model = lgb.train(param, dtrain, ro, valid_sets=[dtrain], verbose_eval=500)
    test_x = test[train_x.columns.values]
    y = model.predict(test_x)
    test['orderType'] = y
    test[['userid', 'orderType']].to_csv('../result/lgb_baseline'+str(version)+'.csv', index=False)

train = pd.read_csv('basis.csv')
test = pd.read_csv('basis.csv')

action_rzc = pd.read_csv('feature_cross.csv')
train = pd.merge(train, action_rzc, on='userid', how='left')
test = pd.merge(test, action_rzc, on='userid', how='left')

ysm = pd.read_csv('ysm.csv')
train = pd.merge(train, ysm, on='userid', how='left')
test = pd.merge(test, ysm, on='userid', how='left')

train.to_csv('../feature/train' + str(version) + '.csv', index=False)
test.to_csv('../feature/test' + str(version) + '.csv', index=False)

XGB(train, test)