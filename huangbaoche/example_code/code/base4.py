# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
version = 5

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

train = pd.read_csv('../feature/train' + str(version) + '.csv')
test = pd.read_csv('../feature/test' + str(version) + '.csv')
XGB(train, test) 