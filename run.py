#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
インターフェース用

事前に以下を実行する必要あり
# ローカルライブラリuser_item_preprocessをビルドする　
> pip install ./src/user_item_preprocess
"""
from datetime import datetime as dt
import os

import pandas as pd
import numpy as np
from sklearn import tree
import graphviz

import user_item_preprocess
from src import util

#############################
# 入力をセット
FILE_MAIN     = './data/20190504_data_analytics/main.csv'
FILE_SUB_USER = './data/20190504_data_analytics/sub_user_status.csv'
FILE_SUB_ITEM = './data/20190504_data_analytics/sub_news_type.csv'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
OUTPUT_DIR = './output'

def main_process(now_datetime='2019-04-25 23:59:59', before_datetime='2019-04-19 00:00:00'):
    '''
    メインプロセス。
    最終的にOUTPUT_DIRに処理済みファイルを出力する。

    ARGUMENT
    --------------
    now_datetime:
        モニタリングしたいタイミング
    before_datetime:
        主に、user_idのstatus変化を計測するために必要。
    '''
    global user_info, get_uid, get_diff_days

    main     = pd.read_csv(FILE_MAIN,     dtype=str).dropna()
    sub_user = pd.read_csv(FILE_SUB_USER, dtype=str).dropna()
    sub_item = pd.read_csv(FILE_SUB_ITEM, dtype=str).dropna()

    ui = user_info(main, sub_user, sub_item)

    result_list = []
    for i,user_id in enumerate(sub_user['user_id'].unique()):
        if i % 100 == 0:
            print('--- now processing at {}/{}'.format(i, len(sub_user['user_id'].unique())))
        user_info = ui.get_user_info(user_id, now_datetime)
        user_info['now_datetime'] = now_datetime
        user_info['now_user_status'] = ui.get_user_status(user_id, now_datetime)
        user_info['before_datetime'] = before_datetime
        user_info['before_user_status'] = ui.get_user_status(user_id, before_datetime)
        result_list.append(user_info)

    df = pd.DataFrame(result_list)

    df['qcut_detail_past_30day_cnt'] = util.labeled_qcut(df['detail_past_30day_cnt'], q=[0.0, 0.80, 0.90, 1.])
    df['qcut_click_past_30day_cnt']  = util.labeled_qcut(df['click_past_30day_cnt'], q=[0.0, 0.80, 0.90, 1.])

    df['cut_diff_days_from_last_start_free'] = util.labeled_cut(df['diff_days_from_last_start_free'], bins=[-0.1, 30, 90, 180, 365, np.inf])
    df['cut_diff_days_from_last_start_paid'] = util.labeled_cut(df['diff_days_from_last_start_paid'], bins=[-0.1, 30, 90, 180, 365, np.inf])

    df['befor_now_user_status'] = df['before_user_status'] + '->' + df['now_user_status']

    # 処理済みの中間ファイルを出力
    output_file_path = os.path.join(OUTPUT_DIR, 'preprocessed_data.csv')
    df.to_csv(output_file_path, sep=',', index=False)

    #######################################
    # --- 決定木の樹形図.png を出力する ---
    cols_click = [
            #'click_past_7day_cnt',
            #'click_past_30day_cnt',
            #'click_past_60day_cnt',
            #'click_past_90day_cnt',
            #'click_past_all_cnt',
            'click_free_past_7day_cnt',
            'click_free_past_30day_cnt',
            'click_free_past_60day_cnt',
            #'click_free_past_90day_cnt',
            #'click_free_past_all_cnt',
            'click_paid_past_7day_cnt',
            'click_paid_past_30day_cnt',
            'click_paid_past_60day_cnt',
            #'click_paid_past_90day_cnt',
            #'click_paid_past_all_cnt',
            ]
    cols_detail = [
            #'detail_past_7day_cnt',
            #'detail_past_30day_cnt',
            #'detail_past_60day_cnt',
            #'detail_past_90day_cnt',
            #'detail_past_all_cnt',
            'detail_free_past_7day_cnt',
            'detail_free_past_30day_cnt',
            'detail_free_past_60day_cnt',
            #'detail_free_past_90day_cnt',
            #'detail_free_past_all_cnt',
            'detail_paid_past_7day_cnt',
            'detail_paid_past_30day_cnt',
            'detail_paid_past_60day_cnt',
            #'detail_paid_past_90day_cnt',
            #'detail_paid_past_all_cnt',
            ]

    ###############################
    # 有料化（'start_free->start_paid'）の決定木分岐
    cols_diff_days = ['diff_days_from_last_start_free']
    X_cols = cols_diff_days + cols_click + cols_detail
    y_cate = ['start_free->start_paid', 'start_free->start_free']
    _df = df.loc[df['befor_now_user_status'].isin(y_cate), :]
    X = _df[X_cols]
    y = (_df['befor_now_user_status']==y_cate[0]).astype(int)
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)
    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=X_cols, class_names=y_cate)
    graph = graphviz.Source(dot_data)
    png_bytes = graph.pipe(format='png')
    output_file_path = os.path.join(OUTPUT_DIR, 'start_free->start_paid.png')
    with open(output_file_path, 'wb') as f:
        f.write(png_bytes)

    ## ランダムフォレストによる変数重要度のスコアリングを行う。
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=250, random_state=0).fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    rank = np.argsort(importances)[::-1]
    importance_result = dict(
            importances_mean = importances,
            importances_std = std,
            rank = rank+1,
            feature_name = X.columns
            )
    output_file_path = os.path.join(OUTPUT_DIR, 'feature_importance__start_free->start_paid.csv')
    pd.DataFrame(importance_result).to_csv(output_file_path, index=False)

    ###############################
    # 有料離脱（'start_paid->end_paid'）の決定木分岐
    cols_diff_days = ['diff_days_from_last_start_paid']
    X_cols = cols_diff_days + cols_click + cols_detail
    y_cate = ['start_paid->end_paid', 'start_paid->start_paid']
    _df = df.loc[df['befor_now_user_status'].isin(y_cate), :]
    X = _df[X_cols]
    y = (_df['befor_now_user_status']==y_cate[0]).astype(int)
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)
    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=X_cols, class_names=y_cate)
    graph = graphviz.Source(dot_data)
    png_bytes = graph.pipe(format='png')
    output_file_path = os.path.join(OUTPUT_DIR, 'start_paid->end_paid.png')
    with open(output_file_path, 'wb') as f:
        f.write(png_bytes)

    ## ランダムフォレストによる変数重要度のスコアリングを行う。
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=250, random_state=0).fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    rank = np.argsort(importances)[::-1]
    importance_result = dict(
            importances_mean = importances,
            importances_std = std,
            rank = rank+1,
            feature_name = X.columns
            )
    output_file_path = os.path.join(OUTPUT_DIR, 'feature_importance__start_paid->end_paid.csv')
    pd.DataFrame(importance_result).to_csv(output_file_path, index=False)





class user_info:
    def __init__(self, main, sub_user, sub_item):
        self.click_uid  = get_uid(main.loc[main['event']=='click', :])
        self.detail_uid = get_uid(main.loc[main['event']=='detail', :])

        _main = main.merge(sub_item, how='left', on='news_id')
        _main['event_news_type'] = _main['event']+'_'+_main['news_type']
        self.click_free_uid  = get_uid(_main.loc[_main['event_news_type']=='click_free', :])
        self.click_paid_uid  = get_uid(_main.loc[_main['event_news_type']=='click_paid', :])
        self.detail_free_uid = get_uid(_main.loc[_main['event_news_type']=='detail_free', :])
        self.detail_paid_uid = get_uid(_main.loc[_main['event_news_type']=='detail_paid', :])

        self.sub_user = sub_user
        self.sub_user['dt_time'] = pd.to_datetime(self.sub_user['time'],
                                               format=DATETIME_FORMAT)

    def get_user_info(self, user_id, datetime, diff_days=[7,30,60,90]):
        '''
        EXAMPLE
        ------------
        user_id  = '4d82760bb00a54316a6f939fd620744a'
        datetime = '2019-04-15 23:59:59'
        self = user_info(main, sub_user, sub_item)
        print( self.get_user_info(user_id, datetime) )
            > """
                {'detail_past_30day_cnt': 3,
                 'detail_past_60day_cnt': 6,
                 'detail_past_7day_cnt': 0,
                 'detail_past_90day_cnt': 9,
                 'detail_past_all_cnt': 12,
                 'diff_days_from_last_end_paid': nan,
                 'diff_days_from_last_start_free': 2035.9763888888901,
                 'diff_days_from_last_start_paid': nan,
                 'user_id': '4d82760bb00a54316a6f939fd620744a'}
              """
        '''
        user_info = dict(user_id=user_id)

        click_info = self.click_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
        click_info = {'click_'+key:val for key,val in click_info.items()}

        detail_info = self.detail_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
        detail_info = {'detail_'+key:val for key,val in detail_info.items()}

        click_free_info = self.click_free_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
        click_free_info = {'click_free_'+key:val for key,val in click_free_info.items()}

        click_paid_info = self.click_free_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
        click_paid_info = {'click_paid_'+key:val for key,val in click_paid_info.items()}

        detail_free_info = self.click_free_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
        detail_free_info = {'detail_free_'+key:val for key,val in detail_free_info.items()}

        detail_paid_info = self.click_free_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
        detail_paid_info = {'detail_paid_'+key:val for key,val in detail_paid_info.items()}

        user_info.update(click_info)
        user_info.update(detail_info)
        user_info.update(click_free_info)
        user_info.update(click_paid_info)
        user_info.update(detail_free_info)
        user_info.update(detail_paid_info)

        _sub_user = self.sub_user.loc[self.sub_user['user_id']==user_id, :]

        def get_last_date(status='start_free'):
            bo_index = _sub_user['status']==status
            return _sub_user.loc[bo_index, 'time'].max()

        last_dt_start_free = get_last_date(status='start_free')
        last_dt_start_paid = get_last_date(status='start_paid')
        last_dt_end_paid   = get_last_date(status='end_paid')

        diff_days_from_last_start_free = np.nan
        diff_days_from_last_start_paid = np.nan
        diff_days_from_last_end_paid   = np.nan

        if last_dt_start_free is not np.nan:
            diff_days_from_last_start_free = get_diff_days(last_dt_start_free, datetime)
        if last_dt_start_paid is not np.nan:
            diff_days_from_last_start_paid = get_diff_days(last_dt_start_paid, datetime)
        if last_dt_end_paid is not np.nan:
            diff_days_from_last_end_paid = get_diff_days(last_dt_end_paid, datetime)

        user_info.update({'diff_days_from_last_start_free':diff_days_from_last_start_free})
        user_info.update({'diff_days_from_last_start_paid':diff_days_from_last_start_paid})
        user_info.update({'diff_days_from_last_end_paid':diff_days_from_last_end_paid})

        return user_info

    def get_user_status(self, user_id, datetime):
        '''
        そのuser_idのそのdatetimeでのstatusを返却します。

        EXAMPLE
        ------------
        user_id  = '4d82760bb00a54316a6f939fd620744a'
        datetime = '2019-04-15 23:59:59'
        self = user_info(main, sub_user, sub_item)
        print( self.get_user_status(user_id, datetime) )
            > 'start_free'
        '''
        dt_datetime = dt.strptime(datetime, DATETIME_FORMAT)
        _sub_user = self.sub_user.loc[self.sub_user['user_id']==user_id ,:]
        bo_index = dt_datetime >= _sub_user['dt_time']
        if bo_index.sum() > 0:
            the_datetime = _sub_user.loc[bo_index, 'dt_time'].max()
            the_status = _sub_user.loc[_sub_user['dt_time']==the_datetime, 'status'].iloc[0]
        else:
            the_status = 'no_record'
        return the_status



def get_uid(main):
    '''
    メインテーブルを処理し、ユーザーのアイテム接触頻度を集計するインスタンスを得る。
    '''
    user_ids  = main.loc[:, 'user_id']
    item_ids  = main.loc[:, 'news_id']
    datetimes = main.loc[:, 'time']
    uid = user_item_preprocess.user_item_datetime.preprocesser(
            user_ids, item_ids, datetimes,
            datetime_format=DATETIME_FORMAT
            )
    return uid

def get_diff_days(from_str_datetime, to_str_datetime):
    '''
    EXAMPLE
    ----------
    from_str_datetime = '2019-04-01 12:00:00'
    to_str_datetime   = '2019-04-03 23:59:59'

    print(diff_days(from_str_datetime, to_str_datetime))
      > 2.4999884259268583
    '''
    total_days = lambda str_date: user_item_preprocess.util.str_to_total_days(str_date, format=DATETIME_FORMAT)
    from_days = total_days(from_str_datetime)
    to_days   = total_days(to_str_datetime)
    return to_days - from_days


if __name__ == '__main__':
    main_process()
