import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
version = 6

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


def getHistoryFeature(df):
    # --get order feature
    feature = df.groupby('userid')['orderType'].agg(['sum', 'count']).reset_index().rename(
        columns={'sum': 'order_num_1', 'count': 'order_num'})

    feature['order_num_0'] = feature['order_num'] - feature['order_num_1']
    feature['order_ratio_0'] = feature['order_num_0'].astype('float') / feature['order_num']
    feature['order_ratio_1'] = feature['order_num_1'].astype('float') / feature['order_num']
    f = feature

    # --get total feature
    feature = df.groupby('userid')['city', 'country', 'continent'].count().reset_index().rename(
        columns={'city': 'city_num', 'country': 'country_num', 'continent': 'continent_num'})

    f = pd.merge(f, feature, on='userid', how='left')
    le = preprocessing.LabelBinarizer()

    encoder1 = le.fit(df.country.values)

    le = preprocessing.LabelBinarizer()
    encoder2 = le.fit(df.continent.values)

    country_encoder = encoder1.transform(df.country.values)

    country_encoder_col = ['country_%d' % i for i in range(country_encoder.shape[1])]
    df1 = pd.DataFrame(country_encoder, columns=country_encoder_col)

    df1['userid'] = df['userid'].values
    feature = df1.groupby('userid')[country_encoder_col].agg(['sum', 'count']).reset_index()
    f = pd.merge(f, feature, on='userid', how='left')

    # --get continent feature
    #     le = preprocessing.LabelBinarizer()
    continent_encoder = encoder2.transform(df.continent.values)
    continent_encoder_col = ['continent_%d' % i for i in range(continent_encoder.shape[1])]
    df1 = pd.DataFrame(continent_encoder, columns=continent_encoder_col)
    df1['userid'] = df['userid'].values
    feature = df1.groupby('userid')[continent_encoder_col].agg(['sum', 'count']).reset_index()

    f = pd.merge(f, feature, on='userid', how='left')

    return f


def getUserProfileFeature(df):
    le = preprocessing.LabelBinarizer()

    encoder = le.fit_transform(df.gender.fillna('_NA_').values)

    encoder_col = ['gender_%d' % i for i in range(encoder.shape[1])]
    df1 = pd.DataFrame(encoder, columns=encoder_col)
    df1['userid'] = df['userid'].values
    f = df1

    le = preprocessing.LabelBinarizer()
    encoder = le.fit_transform(df.province.fillna('_NA_').values)
    encoder_col = ['province_%d' % i for i in range(encoder.shape[1])]
    df1 = pd.DataFrame(encoder, columns=encoder_col)
    df1['userid'] = df['userid'].values
    f = pd.merge(f, df1, on='userid', how='left')

    le = preprocessing.LabelBinarizer()
    encoder = le.fit_transform(df.age.fillna('_NA_').values)
    encoder_col = ['age_%d' % i for i in range(encoder.shape[1])]
    df1 = pd.DataFrame(encoder, columns=encoder_col)
    df1['userid'] = df['userid'].values
    f = pd.merge(f, df1, on='userid', how='left')

    return f


def getCommentFeature(df):
    feature = df.groupby('userid')['rating'].agg(
        ['max', 'min', 'mean', 'sum', 'count', 'median', 'std']).reset_index().rename(
        columns={'max': 'rate_max', 'min': 'rate_min', 'mean': 'rate_mean', 'sum': 'rate_sum', 'count': 'rate_count',
                 'median': 'rate_median', 'std': 'rate_std'})
    return feature


def getActionFeature(df):
    def fea_last(result):
        window = 6

        f = df.groupby('userid').apply(lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(
            drop=True).rename(
            columns={'actionType': 'actionType_last', 'actionTime': 'actionTime_last'})

        f2 = pd.concat([f, f.groupby('userid').rank(method='first', ascending=False).astype('int').reset_index().rename(
            columns={'actionTime_last': 'actionTime_last_rank'})['actionTime_last_rank']], axis=1)

        ff1 = f2.pivot('userid', 'actionTime_last_rank', 'actionType_last')

        # ---get last 6 type diff
        ff1_diff = -ff1.diff(1, axis=1)
        ff1_diff.columns = ['last%d_type_1diff%d' % (window, i) for i in range(ff1_diff.shape[1])]
        ff1_diff = ff1_diff.iloc[:, 1:].reset_index()
        result = pd.merge(result, ff1_diff, on='userid', how='left')

        # ---get last 6 time diff
        ff2 = f2.pivot('userid', 'actionTime_last_rank', 'actionTime_last')
        ff2_diff = -ff2.diff(1, axis=1)
        ff2_diff.columns = ['last%d_time_1diff%d' % (window, i) for i in range(ff2_diff.shape[1])]
        ff2_diff = ff2_diff.iloc[:, 1:]
        result = pd.merge(result, ff2_diff.reset_index(), on='userid', how='left')

        for sub_window in range(2, window + 1):
            result = pd.merge(result,
                              ff1.iloc[:, :sub_window].max(axis=1).reset_index(name='last%d_type_max' % sub_window),
                              on='userid', how='left')
            result = pd.merge(result,
                              ff1.iloc[:, :sub_window].min(axis=1).reset_index(name='last%d_type_min' % sub_window),
                              on='userid', how='left')
            result = pd.merge(result,
                              ff1.iloc[:, :sub_window].mean(axis=1).reset_index(name='last%d_type_mean' % sub_window),
                              on='userid', how='left')
            result = pd.merge(result, ff1.iloc[:, :sub_window].median(axis=1).reset_index(
                name='last%d_type_median' % sub_window), on='userid', how='left')
            result = pd.merge(result,
                              ff1.iloc[:, :sub_window].std(axis=1).reset_index(name='last%d_type_std' % sub_window),
                              on='userid', how='left')
            result = pd.merge(result,
                              ff1.iloc[:, :sub_window].sum(axis=1).reset_index(name='last%d_type_sum' % sub_window),
                              on='userid', how='left')

        for sub_window in range(2, window):
            result = pd.merge(result, ff2_diff.iloc[:, :sub_window].max(axis=1).reset_index(
                name='last%d_time_diff_max' % sub_window), on='userid', how='left')
            result = pd.merge(result, ff2_diff.iloc[:, :sub_window].min(axis=1).reset_index(
                name='last%d_time_diff_min' % sub_window), on='userid', how='left')
            result = pd.merge(result, ff2_diff.iloc[:, :sub_window].mean(axis=1).reset_index(
                name='last%d_time_diff_mean' % sub_window), on='userid', how='left')
            result = pd.merge(result, ff2_diff.iloc[:, :sub_window].median(axis=1).reset_index(
                name='last%d_time_diff_median' % sub_window), on='userid', how='left')
            result = pd.merge(result, ff2_diff.iloc[:, :sub_window].std(axis=1).reset_index(
                name='last%d_time_diff_std' % sub_window), on='userid', how='left')
            result = pd.merge(result, ff2_diff.iloc[:, :sub_window].sum(axis=1).reset_index(
                name='last%d_time_diff_sum' % sub_window), on='userid', how='left')

        # ---get last 6 type
        ff1.columns = ['last%d_type_index%d' % (window, i + 1) for i in range(ff1.shape[1])]
        ff1 = ff1.reset_index()
        result = pd.merge(result, ff1, on='userid', how='left')

        for sub_window in range(1, window):
            f = df.groupby('userid').nth([-i for i in range(1, sub_window + 1)]).reset_index().rename(
                columns={'actionType': 'actionType_last', 'actionTime': 'actionTime_last'})
            ff = f.groupby(['userid', 'actionType_last'])['actionTime_last'].count().reset_index().rename(
                columns={'actionTime_last': 'actionNum_last'}).pivot('userid', 'actionType_last', 'actionNum_last')
            feature = ff.apply(lambda x: x / np.sum(x))
            feature.columns = ['last%d_action_type_num_tol_pct%d' % (sub_window, i + 1) for i in range(ff.shape[1])]
            feature = feature.reset_index()
            result = pd.merge(result, feature, on='userid', how='left')

            feature = ff.apply(lambda x: x / np.sum(x), axis=1)
            feature.columns = ['last%d_action_type_num_person_pct%d' % (sub_window, i + 1) for i in range(ff.shape[1])]
            feature = feature.reset_index()
            result = pd.merge(result, feature, on='userid', how='left')

        return result

    # ---sort every type last 1 action feature
    def fea_every_type_last_one(result):
        f = df.groupby(['userid', 'actionType']).apply(
            lambda x: x.sort_values('actionTime', ascending=False).head(1)).reset_index(drop=True).rename(
            columns={'actionTime': 'type_actionTime_last'})

        ff3 = f.pivot('userid', 'actionType', 'type_actionTime_last')
        for periods in range(1,7):
            ff3_diff = ff3.diff(periods, axis=1)
            ff3_diff.columns = ['type_%d_lsttime_%ddiff' % (i,periods) for i in range(ff3_diff.shape[1])]
            ff3_diff = ff3_diff.iloc[:, periods:].reset_index()
            result = pd.merge(result, ff3_diff, on='userid', how='left')

        ff3.columns = ['type_%d_lsttime' % (i + 1) for i in range(ff3.shape[1])]
        ff3 = ff3.reset_index()
        result = pd.merge(result, ff3, on='userid', how='left')
        return result

    def fea_every_type(result):
        window = 5
        for t in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            df_type = df[df.actionType == t]
            f = df_type.groupby('userid').apply(
                lambda x: x.sort_values('actionTime', ascending=False).head(window)).reset_index(drop=True).rename(
                columns={'actionTime': 'actionTime_last'})
            f2 = pd.concat(
                [f, f.groupby('userid').rank(method='first', ascending=False).astype('int').reset_index().rename(
                    columns={'actionTime_last': 'actionTime_last_rank'})['actionTime_last_rank']], axis=1)

            ff1 = f2.pivot('userid', 'actionTime_last_rank', 'actionTime_last')
            for periods in range(1,5):
                ff1_diff = -ff1.diff(periods, axis=1)
                ff1_diff.columns = ['last%d_type%d_time_%ddiff%d' % (window, t, periods, i) for i in range(ff1_diff.shape[1])]
                ff1_diff = ff1_diff.iloc[:, periods:]

                for sub_window in range(2,window-periods+1):
                    result = pd.merge(result,
                                      ff1_diff.iloc[:,:sub_window].max(axis=1).reset_index(name='last%d_sub%d_type%d_time_%ddiff_max' % (window, sub_window, t,periods)),
                                      on='userid', how='left')

                    result = pd.merge(result,
                                      ff1_diff.iloc[:,:sub_window].min(axis=1).reset_index(name='last%d_sub%d_type%d_time_%ddiff_min' % (window,sub_window, t,periods)),
                                      on='userid', how='left')

                    result = pd.merge(result,
                                      ff1_diff.iloc[:,:sub_window].mean(axis=1).reset_index(name='last%d_sub%d_type%d_time_%ddiff_mean' % (window,sub_window, t,periods)),
                                      on='userid', how='left')

                    result = pd.merge(result,
                                      ff1_diff.iloc[:,:sub_window].median(axis=1).reset_index(name='last%d_sub%d_type%d_time_%ddiff_median' % (window,sub_window, t,periods)),
                                      on='userid', how='left')

                    result = pd.merge(result,
                                      ff1_diff.iloc[:,:sub_window].std(axis=1).reset_index(name='last%d_sub%d_type%d_time_%ddiff_std' % (window,sub_window, t,periods)),
                                      on='userid', how='left')

                    result = pd.merge(result,
                                      ff1_diff.iloc[:,:sub_window].sum(axis=1).reset_index(name='last%d_sub%d_type%d_time_%ddiff_sum' % (window,sub_window, t,periods)),
                                      on='userid', how='left')

                result = pd.merge(result, ff1_diff.reset_index(), on='userid', how='left')

            ff1.columns = ['last%d_type%d_time%d' % (window, t, i + 1) for i in range(ff1.shape[1])]
            ff1 = ff1.reset_index()
            result = pd.merge(result, ff1, on='userid', how='left')

        return result

    def fea_every_type_last_distance(result):
        # ---the distance from last type to type n
        def action_distance_n(group, n):
            try:
                return list(group)[::-1].index(n)
            except ValueError:
                return np.nan

        f = df.groupby('userid', sort=False)['actionType']
        for t in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            result = pd.merge(result, f.apply(action_distance_n, t).reset_index(name='type%d_distance' % t),
                              on='userid', how='left')

        return result

    def fea_last_type_n_to_end_time(result):
        ff = df.groupby('userid').apply(lambda x: x.drop_duplicates(subset='actionType',keep='last')).reset_index(drop=True)
        ffp = ff.pivot('userid', 'actionType', 'actionTime')
        last = df.groupby('userid').nth(-1)

        feature = pd.concat([(last['actionTime'] - ffp.iloc[:, i]) for i in range(ffp.shape[1])],axis=1)
        feature.columns = [(('last_type%d_to_end_time' % (i + 1))) for i in range(feature.shape[1])]
        feature = feature.reset_index()
        result = pd.merge(result, feature, on='userid', how='left')
        return result

    def fea_type5_to_next_type6(result):
        def groupby_apply(group):
            sel = []
            b = False
            label = []
            cur = -1
            for type in group['actionType']:
                if not b and type == 5:
                    b = True
                    sel.append(b)
                    label.append(cur)
                elif b and type == 6:
                    sel.append(b)
                    label.append(cur)
                    b = False
                    cur = cur - 1
                else:
                    sel.append(b)
                    label.append(cur)
            group['type5_to_next_type6_last_n'] = label
            group['type5_to_next_type6_last_n'] = group['type5_to_next_type6_last_n'] - cur
            rs = group[sel]

            return rs

        ff = df.groupby('userid').apply(groupby_apply).reset_index(drop=True)

        ff1 = ff.groupby(['userid', 'type5_to_next_type6_last_n'])['actionTime'].nth(-1) - \
              ff.groupby(['userid', 'type5_to_next_type6_last_n'])['actionTime'].nth(0)
        ffp = (ff1.reset_index(name='actionTimeDiff')).pivot('userid', 'type5_to_next_type6_last_n', 'actionTimeDiff')
        feature = ffp.iloc[:, 0:21]

        feature.columns = [(('type5_to_next_type6_last%d' % (i))) for i in feature.columns]
        feature = feature.reset_index()
        result = pd.merge(result, feature, on='userid', how='left')

        ff_diff = ff.groupby(['userid', 'type5_to_next_type6_last_n'])['actionTime'].diff()
        ff['diff'] = ff_diff

        ff_diff_group = ff.groupby(['userid','type5_to_next_type6_last_n'])['diff']

        def common_func(result,func,funcname):
            feature = func().reset_index(name='name').pivot('userid', 'type5_to_next_type6_last_n', 'name')
            feature = feature.iloc[:, 0:21]
            feature.columns = [(('type5_to_next_type6_diff_'+funcname+'_last%d' % (i))) for i in feature.columns]
            feature = feature.reset_index()
            result = pd.merge(result, feature, on='userid', how='left')
            return result

        result = common_func(result,ff_diff_group.max,'max')
        result = common_func(result, ff_diff_group.min, 'min')
        result = common_func(result, ff_diff_group.mean, 'mean')
        result = common_func(result, ff_diff_group.median, 'median')
        result = common_func(result, ff_diff_group.std, 'std')
        result = common_func(result, ff_diff_group.sum, 'sum')

        return result

    result = commonGetActionFeature(df, 'total_action')
    result = fea_last(result)
    result = fea_every_type_last_one(result)
    result = fea_every_type(result)
    result = fea_every_type_last_distance(result)
    result = fea_last_type_n_to_end_time(result)
    result = fea_type5_to_next_type6(result)
    return result


def getPredictOrderActionFeature(action, order_history):
    predict_order_time = order_history.groupby('userid', as_index=False, sort=False)['orderTime'].max()
    predict_order_action = pd.merge(action, predict_order_time, how='left', on='userid')
    df = predict_order_action.loc[(pd.isna(predict_order_action.orderTime)) | (
        predict_order_action.actionTime > predict_order_action.orderTime + 30), :]
    df = df.drop('orderTime', axis=1)
    df = df.reset_index(drop=True)

    result = commonGetActionFeature(df, 'predict_order')

    return result


def getDiffLessDayActionFeature(df):
    dif = df['actionTime'].diff()
    df['dif'] = dif

    def last_diff_less_day(group):
        try:
            index = list(group['dif'] > 86400)[::-1].index(True)
            return group.iloc[len(group) - index - 1:, :]
        except:
            return group

    f = df.groupby('userid').apply(last_diff_less_day).reset_index(drop=True).drop('dif', axis=1)

    result = commonGetActionFeature(f, 'less_day')

    def fea_first_type_n_to_end_time(result):
        ff = f.groupby('userid').apply(lambda x: x.drop_duplicates(subset='actionType')).reset_index(drop=True)
        ffp = ff.pivot('userid', 'actionType', 'actionTime')
        last = f.groupby('userid').nth(-1)

        feature = pd.concat([(last['actionTime'] - ffp.iloc[:, i]) for i in range(ffp.shape[1])],axis=1)
        feature.columns = [('less_day' + ('_first_type%d_to_end_time' % (i + 1))) for i in range(feature.shape[1])]
        feature = feature.reset_index()
        result = pd.merge(result, feature, on='userid', how='left')
        return result

    result = fea_first_type_n_to_end_time(result)

    return result


def commonGetActionFeature(df, prefix):
    df = df.reset_index(drop=True)

    def action_diff_statistic(result):
        df_diff = df.groupby('userid')['actionTime', 'actionType'].diff().rename(
            columns={'actionTime': 'actionTime_diff', 'actionType': 'actionType_diff'})
        df_diff['userid'] = df['userid']

        df_diff_group = df_diff.groupby('userid')

        feature = df_diff_group.max().reset_index().rename(
            columns={'actionTime_diff': prefix + '_time_diff_max', 'actionType_diff': prefix + '_type_diff_max'})
        result = pd.merge(result, feature, on='userid', how='left')

        feature = df_diff_group.min().reset_index().rename(
            columns={'actionTime_diff': prefix + '_time_diff_min', 'actionType_diff': prefix + '_type_diff_min'})
        result = pd.merge(result, feature, on='userid', how='left')

        feature = df_diff_group.mean().reset_index().rename(
            columns={'actionTime_diff': prefix + '_time_diff_mean', 'actionType_diff': prefix + '_type_diff_mean'})
        result = pd.merge(result, feature, on='userid', how='left')

        feature = df_diff_group.median().reset_index().rename(
            columns={'actionTime_diff': prefix + '_time_diff_median',
                     'actionType_diff': prefix + '_type_diff_median'})
        result = pd.merge(result, feature, on='userid', how='left')

        feature = df_diff_group.std().reset_index().rename(
            columns={'actionTime_diff': prefix + '_time_diff_std',
                     'actionType_diff': prefix + '_type_diff_std'})
        result = pd.merge(result, feature, on='userid', how='left')

        feature = df_diff_group.sum().reset_index().rename(
            columns={'actionTime_diff': prefix + '_time_diff_sum',
                     'actionType_diff': prefix + '_type_diff_sum'})
        result = pd.merge(result, feature, on='userid', how='left')
        return result

    def action_type_statistic(result):
        df_group_type = df.groupby('userid')['actionType']
        feature = df_group_type.max().reset_index(name=prefix + '_type_max')
        result = pd.merge(result, feature, on='userid', how='left')
        feature = df_group_type.min().reset_index(name=prefix + '_type_min')
        result = pd.merge(result, feature, on='userid', how='left')
        feature = df_group_type.mean().reset_index(name=prefix + '_type_mean')
        result = pd.merge(result, feature, on='userid', how='left')
        feature = df_group_type.median().reset_index(name=prefix + '_type_median')
        result = pd.merge(result, feature, on='userid', how='left')
        feature = df_group_type.std().reset_index(name=prefix + '_type_std')
        result = pd.merge(result, feature, on='userid', how='left')
        feature = df_group_type.sum().reset_index(name=prefix + '_type_sum')
        result = pd.merge(result, feature, on='userid', how='left')

        return result

    def action_num(result):
        feature_action_num = df.groupby(['userid', 'actionType'])['actionTime'].count().reset_index().rename(
            columns={'actionTime': 'actionNum'}).pivot('userid', 'actionType', 'actionNum')
        feature = feature_action_num.apply(lambda x: x / np.sum(x))
        feature.columns = [(prefix + ('_action_type_num_tol_pct%d' % (i + 1))) for i in range(feature.shape[1])]
        feature = feature.reset_index()
        result = pd.merge(result, feature, on='userid', how='left')

        feature = feature_action_num.apply(lambda x: x / np.sum(x), axis=1)
        feature.columns = [(prefix + ('_action_type_num_person_pct%d' % (i + 1))) for i in range(feature.shape[1])]
        feature = feature.reset_index()
        result = pd.merge(result, feature, on='userid', how='left')
        return result

    def every_type_fsttime_to_lsttime(result):
        f1 = df.groupby(['userid', 'actionType']).nth(0).reset_index().rename(
            columns={'actionTime': 'type_actionTime_first'})

        fp1 = f1.pivot('userid','actionType','type_actionTime_first')

        f2 = df.groupby(['userid', 'actionType']).nth(-1).reset_index().rename(
            columns={'actionTime': 'type_actionTime_last'})
        fp2 = f2.pivot('userid','actionType','type_actionTime_last')
        ff = fp2-fp1
        ff.columns = [(prefix + ('_type%d_lsttime-fsttime' % (i))) for i in ff.columns]
        ff = ff.reset_index()
        result = pd.merge(result, ff, on='userid', how='left')
        return result

    def fea_every_type_first_one(result):
        f = df.groupby(['userid', 'actionType']).nth(0).reset_index().rename(columns={'actionTime': 'type_actionTime_first'})

        ff3 = f.pivot('userid', 'actionType', 'type_actionTime_first')
        for periods in range(1,5):
            ff3_diff = ff3.diff(periods, axis=1)
            ff3_diff.columns = [(prefix + ('_type_%d_fsttime_%ddiff' % (i,periods))) for i in range(ff3_diff.shape[1])]
            ff3_diff = ff3_diff.iloc[:, periods:].reset_index()
            result = pd.merge(result, ff3_diff, on='userid', how='left')

        ff3.columns = [prefix + '_type_%d_fsttime' % (i + 1) for i in ff3.columns]
        ff3 = ff3.reset_index()
        result = pd.merge(result, ff3, on='userid', how='left')
        return result

    def fea_type5_to_next_type6_statistics(result):
        def groupby_apply(group):
            sel = []
            b = False
            label = []
            cur = -1
            for type in group['actionType']:
                if not b and type == 5:
                    b = True
                    sel.append(b)
                    label.append(cur)
                elif b and type == 6:
                    sel.append(b)
                    label.append(cur)
                    b = False
                    cur = cur - 1
                else:
                    sel.append(b)
                    label.append(cur)
            group['type5_to_next_type6_last_n'] = label
            group['type5_to_next_type6_last_n'] = group['type5_to_next_type6_last_n'] - cur
            rs = group[sel]
            return rs

        ff = df.groupby('userid').apply(groupby_apply).reset_index(drop=True)

        ff1 = ff.groupby(['userid', 'type5_to_next_type6_last_n'])['actionTime'].nth(-1) - \
              ff.groupby(['userid', 'type5_to_next_type6_last_n'])['actionTime'].nth(0)
        ff1_groupby = ff1.groupby('userid')

        def common_func(result,func,funcname,prefix):
            feature = func().reset_index(name=prefix+'_type5_to_next_type6_'+funcname)
            result = pd.merge(result, feature, on='userid', how='left')
            return result

        result = common_func(result,ff1_groupby.max,'max',prefix)
        result = common_func(result, ff1_groupby.min, 'min',prefix)
        result = common_func(result, ff1_groupby.mean, 'mean',prefix)
        result = common_func(result, ff1_groupby.median, 'median',prefix)
        result = common_func(result, ff1_groupby.std, 'std',prefix)
        result = common_func(result, ff1_groupby.sum, 'sum',prefix)

        return result

    result = df.drop_duplicates(subset='userid')[['userid']]
    result = action_diff_statistic(result)
    result = action_type_statistic(result)
    result = action_num(result)
    result = every_type_fsttime_to_lsttime(result)
    result = fea_every_type_first_one(result)
    result = fea_type5_to_next_type6_statistics(result)
    return result

def feature_cross(df):
    result = df[['userid']]
    result['5*type_5_lsttime_1diff+6*type_6_lsttime_1diff'] = 5*df['type_5_lsttime_1diff'] + 6*df['type_6_lsttime_1diff']
    result['5*less_day_type_5_fsttime_1diff+6*less_day_type_6_fsttime_1diff'] = 5*df['less_day_type_5_fsttime_1diff'] + 6*df['less_day_type_6_fsttime_1diff']
    result['type_4_lsttime_4diff+5*type_5_lsttime_1diff'] = df['type_4_lsttime_4diff'] + 5*df['type_5_lsttime_1diff']
    result['less_day_type_4_fsttime_4diff+5*less_day_type_5_fsttime_1diff'] = df['less_day_type_4_fsttime_4diff'] + 5*df['less_day_type_5_fsttime_1diff']

    result['type_4_lsttime_1diff+less_day_type_4_fsttime_1diff'] = df['type_4_lsttime_1diff'] + df['less_day_type_4_fsttime_1diff']
    result['type_5_lsttime_1diff+less_day_type_5_fsttime_1diff'] = df['type_5_lsttime_1diff'] + df['less_day_type_5_fsttime_1diff']
    result['type_6_lsttime_1diff+less_day_type_6_fsttime_1diff'] = df['type_6_lsttime_1diff'] + df['less_day_type_6_fsttime_1diff']

    result['2*last6_time_1diff2+3*last6_time_1diff3'] = 2*df['last6_time_1diff2'] + 3*df['last6_time_1diff3']

    result['1*last6_time_1diff1+2*last6_time_1diff2'] = 1*df['last6_time_1diff1'] + 2*df['last6_time_1diff2']

    result['1*last5_type5_time_1diff1+2*last5_type5_time_1diff2'] = 1*df['last5_type5_time_1diff1'] + 2*df['last5_type5_time_1diff2']

    result['type_5_lsttime_1diff+type_6_lsttime_1diff'] = df['type_5_lsttime_1diff'] + df['type_6_lsttime_1diff']
    result['type_5_lsttime_1diff-type_6_lsttime_1diff'] = df['type_5_lsttime_1diff'] - df['type_6_lsttime_1diff']
    result['type_5_lsttime_1diff*type_6_lsttime_1diff'] = df['type_5_lsttime_1diff'] * df['type_6_lsttime_1diff']

    result['last5_type5_time_1diff1+type_5_lsttime_1diff'] = df['last5_type5_time_1diff1'] + df['type_5_lsttime_1diff']
    result['last5_type5_time_1diff1-type_5_lsttime_1diff'] = df['last5_type5_time_1diff1'] - df['type_5_lsttime_1diff']
    result['last5_type5_time_1diff1*type_5_lsttime_1diff'] = df['last5_type5_time_1diff1'] * df['type_5_lsttime_1diff']


    result['last5_type6_time_1diff1+less_day_type_6_fsttime_1diff'] = df['last5_type6_time_1diff1'] + df['less_day_type_6_fsttime_1diff']
    result['last5_type6_time_1diff1-less_day_type_6_fsttime_1diff'] = df['last5_type6_time_1diff1'] - df['less_day_type_6_fsttime_1diff']
    result['last5_type6_time_1diff1*less_day_type_6_fsttime_1diff'] = df['last5_type6_time_1diff1'] * df['less_day_type_6_fsttime_1diff']

    result['last5_type6_time_1diff1+type_6_lsttime_1diff'] = df['last5_type6_time_1diff1'] + df['type_6_lsttime_1diff']
    result['last5_type6_time_1diff1-type_6_lsttime_1diff'] = df['last5_type6_time_1diff1'] - df['type_6_lsttime_1diff']
    result['last5_type6_time_1diff1*type_6_lsttime_1diff'] = df['last5_type6_time_1diff1'] * df['type_6_lsttime_1diff']

    result['last5_type6_time_1diff1+less_day_type_6_fsttime_1diff'] = df['last5_type6_time_1diff1'] + df['less_day_type_6_fsttime_1diff']
    result['last5_type6_time_1diff1-less_day_type_6_fsttime_1diff'] = df['last5_type6_time_1diff1'] - df['less_day_type_6_fsttime_1diff']
    result['last5_type6_time_1diff1*less_day_type_6_fsttime_1diff'] = df['last5_type6_time_1diff1'] * df['less_day_type_6_fsttime_1diff']



    result['order_num_1*type_5_lsttime_1diff'] = df['order_num_1'] * df['type_5_lsttime_1diff']
    result['order_num_1/type_5_lsttime_1diff'] =  df['order_num_1'] / df['type_5_lsttime_1diff']

    result['order_num_1*less_day_type_5_fsttime_1diff'] = df['order_num_1'] * df['less_day_type_5_fsttime_1diff']
    result['order_num_1/less_day_type_5_fsttime_1diff'] = df['order_num_1'] / df['less_day_type_5_fsttime_1diff']

    result['order_num_1*type_6_lsttime_1diff'] = df['order_num_1'] * df['type_6_lsttime_1diff']
    result['order_num_1/type_6_lsttime_1diff'] = df['order_num_1'] / df['type_6_lsttime_1diff']

    result['order_num_1*less_day_type_6_fsttime_1diff'] = df['order_num_1'] * df['less_day_type_6_fsttime_1diff']
    result['order_num_1/less_day_type_6_fsttime_1diff'] = df['order_num_1'] / df['less_day_type_6_fsttime_1diff']

    result['order_num_1*predict_order_action_type_num_person_pct6'] = df['order_num_1'] * df['predict_order_action_type_num_person_pct6']
    result['order_num_1/predict_order_action_type_num_person_pct6'] = df['order_num_1'] / df['predict_order_action_type_num_person_pct6']

    result['order_num_1*predict_order_action_type_num_person_pct5'] = df['order_num_1'] * df['predict_order_action_type_num_person_pct5']
    result['order_num_1/predict_order_action_type_num_person_pct5'] = df['order_num_1'] / df['predict_order_action_type_num_person_pct5']

    result['less_day_time_diff_median-less_day_time_diff_min'] = df['less_day_time_diff_median']-df['less_day_time_diff_min']
    result['less_day_time_diff_median+less_day_time_diff_min'] = df['less_day_time_diff_median']+df['less_day_time_diff_min']
    result['less_day_time_diff_max-less_day_time_diff_median'] = df['less_day_time_diff_max']-df['less_day_time_diff_median']
    result['less_day_time_diff_max+less_day_time_diff_median'] = df['less_day_time_diff_max']+df['less_day_time_diff_median']
    result['less_day_time_diff_max-less_day_time_diff_min'] = df['less_day_time_diff_max']-df['less_day_time_diff_min']
    result['less_day_time_diff_max+less_day_time_diff_min'] = df['less_day_time_diff_max']+df['less_day_time_diff_min']
    result['less_day_time_diff_median-less_day_time_diff_mean'] = df['less_day_time_diff_max']-df['less_day_time_diff_min']
    result['less_day_time_diff_median+less_day_time_diff_mean'] = df['less_day_time_diff_max']+df['less_day_time_diff_min']
    result['less_day_time_diff_max-less_day_time_diff_mean'] = df['less_day_time_diff_max']-df['less_day_time_diff_mean']
    result['less_day_time_diff_max+less_day_time_diff_mean'] = df['less_day_time_diff_max']+df['less_day_time_diff_mean']
    result['less_day_time_diff_min-less_day_time_diff_mean'] = df['less_day_time_diff_min']-df['less_day_time_diff_mean']
    result['less_day_time_diff_min+less_day_time_diff_mean'] = df['less_day_time_diff_min']+df['less_day_time_diff_mean']
    result['less_day_time_diff_mean*less_day_time_diff_std'] = df['less_day_time_diff_mean']*df['less_day_time_diff_std']

    result['predict_order_time_diff_median-predict_order_time_diff_min'] = df['predict_order_time_diff_median'] - df[
        'predict_order_time_diff_min']
    result['predict_order_time_diff_median+predict_order_time_diff_min'] = df['predict_order_time_diff_median'] + df[
        'predict_order_time_diff_min']
    result['predict_order_time_diff_max-predict_order_time_diff_median'] = df['predict_order_time_diff_max'] - df[
        'predict_order_time_diff_median']
    result['predict_order_time_diff_max+predict_order_time_diff_median'] = df['predict_order_time_diff_max'] + df[
        'predict_order_time_diff_median']
    result['predict_order_time_diff_max-predict_order_time_diff_min'] = df['predict_order_time_diff_max'] - df['predict_order_time_diff_min']
    result['predict_order_time_diff_max+predict_order_time_diff_min'] = df['predict_order_time_diff_max'] + df['predict_order_time_diff_min']
    result['predict_order_time_diff_median-predict_order_time_diff_mean'] = df['predict_order_time_diff_max'] - df[
        'predict_order_time_diff_min']
    result['predict_order_time_diff_median+predict_order_time_diff_mean'] = df['predict_order_time_diff_max'] + df[
        'predict_order_time_diff_min']
    result['predict_order_time_diff_max-predict_order_time_diff_mean'] = df['predict_order_time_diff_max'] - df['predict_order_time_diff_mean']
    result['predict_order_time_diff_max+predict_order_time_diff_mean'] = df['predict_order_time_diff_max'] + df['predict_order_time_diff_mean']
    result['predict_order_time_diff_min-predict_order_time_diff_mean'] = df['predict_order_time_diff_min'] - df['predict_order_time_diff_mean']
    result['predict_order_time_diff_min+predict_order_time_diff_mean'] = df['predict_order_time_diff_min'] + df['predict_order_time_diff_mean']
    result['predict_order_time_diff_mean*predict_order_time_diff_std'] = df['predict_order_time_diff_mean'] * df['predict_order_time_diff_std']

    result['less_day_action_type_num_person_pct5*predict_order_action_type_num_person_pct5'] =  df['less_day_action_type_num_person_pct5']*df['predict_order_action_type_num_person_pct5']
    result['less_day_action_type_num_person_pct6*predict_order_action_type_num_person_pct6'] =  df['less_day_action_type_num_person_pct6']*df['predict_order_action_type_num_person_pct6']

    result['predict_order_action_type_num_person_pct5+predict_order_action_type_num_person_pct6'] =  df['predict_order_action_type_num_person_pct5']+df['predict_order_action_type_num_person_pct6']
    result['predict_order_action_type_num_person_pct5*predict_order_action_type_num_person_pct6'] =  df['predict_order_action_type_num_person_pct5']*df['predict_order_action_type_num_person_pct6']

    result['less_day_action_type_num_person_pct5+less_day_action_type_num_person_pct6'] = df['less_day_action_type_num_person_pct5'] + df['less_day_action_type_num_person_pct6']
    result['less_day_action_type_num_person_pct5*less_day_action_type_num_person_pct6'] = df['less_day_action_type_num_person_pct5'] * df['less_day_action_type_num_person_pct6']


    return result

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

history = getHistoryFeature(pd.concat([orderHistory_train, orderHistory_test]))
train = pd.merge(orderFuture_train, history, on='userid', how='left')
test = pd.merge(orderFuture_test, history, on='userid', how='left')

profile = getUserProfileFeature(pd.concat([userProfile_train, userProfile_test]))
train = pd.merge(train, profile, on='userid', how='left')
test = pd.merge(test, profile, on='userid', how='left')

comment = getCommentFeature(pd.concat([userComment_train, userComment_test]))
train = pd.merge(train, comment, on='userid', how='left')
test = pd.merge(test, comment, on='userid', how='left')

action = getActionFeature(pd.concat([action_train, action_test]))
train = pd.merge(train, action, on='userid', how='left')
test = pd.merge(test, action, on='userid', how='left')

predict_order_action = getPredictOrderActionFeature(pd.concat([action_train, action_test]),
                                                   pd.concat([orderHistory_train, orderHistory_test]))
train = pd.merge(train, predict_order_action, on='userid', how='left')
test = pd.merge(test, predict_order_action, on='userid', how='left')

diff_less_hour_action = getDiffLessDayActionFeature(pd.concat([action_train, action_test]))
train = pd.merge(train, diff_less_hour_action, on='userid', how='left')
test = pd.merge(test, diff_less_hour_action, on='userid', how='left')

train.to_csv('../feature/train_uncross' + str(version) + '.csv', index=False)
test.to_csv('../feature/test_uncross' + str(version) + '.csv', index=False)

feature_cross = feature_cross(pd.concat([train,test]))
train = pd.merge(train, feature_cross, on='userid', how='left')
test = pd.merge(test, feature_cross, on='userid', how='left')

train.to_csv('../feature/train' + str(version) + '.csv', index=False)
test.to_csv('../feature/test' + str(version) + '.csv', index=False)

XGB(train, test)
