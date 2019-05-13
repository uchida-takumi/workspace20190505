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

from multiprocessing import Pool

import user_item_preprocess
from src import util

#############################
# 入力をセット
FILE_MAIN     = './data/20190504_data_analytics/main.csv'
FILE_SUB_USER = './data/20190504_data_analytics/sub_user_status.csv'
FILE_SUB_ITEM = './data/20190504_data_analytics/sub_news_type.csv'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
OUTPUT_DIR = './output'

main     = pd.read_csv(FILE_MAIN,     dtype=str).dropna()
sub_user = pd.read_csv(FILE_SUB_USER, dtype=str).dropna()
sub_item = pd.read_csv(FILE_SUB_ITEM, dtype=str).dropna()

# main のデータが大きすぎる場合はユーザーを乱数抽出する。
size = int(main['user_id'].nunique() * 0.25)
_random_user_ids = np.random.choice(main['user_id'].unique(), size=size, replace=False)
main = main.loc[main['user_id'].isin(_random_user_ids), :]

now_datetime='2019-04-25 23:59:59'; before_datetime='2019-04-19 00:00:00'


def preprocess(key_args):
    '''
    key_args = dict(
        user_ids=['ed7bd5bccea3317f7fc60813a25363a2', '2af2427736eae213ba1753888b635d9b'],
        now_datetime='2019-04-25 23:59:59',
        before_datetime='2019-04-19 00:00:00',
    )
    '''
    global main, sub_user, sub_item
    global user_info, get_uid, get_diff_days
    
    user_ids        = key_args['user_ids']
    now_datetime    = key_args['now_datetime']
    before_datetime = key_args['before_datetime']
    
    _main     = main.loc[main.user_id.isin(user_ids), :]
    _sub_user = sub_user.loc[sub_user.user_id.isin(user_ids), :]
    _sub_item = sub_item.copy()
    ui = user_info_class(_main, _sub_user, _sub_item)
    del(_main, _sub_user, _sub_item) # メモリ効率化

    result_list = []
    for i,user_id in enumerate(user_ids):
        if i % 1000 == 0:
            print('--- at {} processing vim{}/{}'.format(dt.now(), i, len(user_ids)))
        user_info = ui.get_user_info(user_id, now_datetime)
        user_info['now_datetime'] = now_datetime
        user_info['now_user_status'] = ui.get_user_status(user_id, now_datetime)
        user_info['before_datetime'] = before_datetime
        user_info['before_user_status'] = ui.get_user_status(user_id, before_datetime)
        result_list.append(user_info)

    return pd.DataFrame(result_list)

def multi(list_of_key_args, n_job=None):
    global preprocess
    p = Pool(n_job)
    result = p.map(preprocess, list_of_key_args)
    p.close()
    return result



def main_process(df):
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
    global now_datetime, before_datetime
    global user_info, get_uid, get_diff_days, multi

    df['detail_past_30day_cnt'] = df['detail_free_past_30day_cnt'] + df['detail_paid_past_30day_cnt']
    df['click_past_30day_cnt'] = df['click_free_past_30day_cnt'] + df['click_paid_past_30day_cnt']
    try:
        _q = [0.0, 0.80, 0.90, 1.]
        df['qcut_detail_past_30day_cnt'] = util.labeled_qcut(df['detail_past_30day_cnt'], q=_q)
        df['qcut_click_past_30day_cnt']  = util.labeled_qcut(df['click_past_30day_cnt'], q=_q)
    except:
        try:
            _q = [0.0, 0.90, 0.95, 1.]
            df['qcut_detail_past_30day_cnt'] = util.labeled_qcut(df['detail_past_30day_cnt'], q=_q)
            df['qcut_click_past_30day_cnt']  = util.labeled_qcut(df['click_past_30day_cnt'], q=_q)
        except:
            _q = [0.0, 0.95, 0.975, 1.]
            df['qcut_detail_past_30day_cnt'] = util.labeled_qcut(df['detail_past_30day_cnt'], q=_q)
            df['qcut_click_past_30day_cnt']  = util.labeled_qcut(df['click_past_30day_cnt'], q=_q)
            

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
    std = np.std(np.array([tree.feature_importances_ for tree in forest.estimators_]),
                 axis=0)
    rank_dict = {index:rank+1 for rank,index in enumerate(np.argsort(-1*importances))}
    importance_result = dict(
            importances_mean = importances,
            importances_std = std,
            rank = [rank_dict[i] for i in range(importances.size)],
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
    rank_dict = {index:rank+1 for rank,index in enumerate(np.argsort(-1*importances))}
    importance_result = dict(
            importances_mean = importances,
            importances_std = std,
            rank = [rank_dict[i] for i in range(importances.size)],
            feature_name = X.columns
            )
    output_file_path = os.path.join(OUTPUT_DIR, 'feature_importance__start_paid->end_paid.csv')
    pd.DataFrame(importance_result).to_csv(output_file_path, index=False)





class user_info_class:
    def __init__(self, main, sub_user, sub_item):
        #self.click_uid  = get_uid(main.loc[main['event']=='click', :])
        #self.detail_uid = get_uid(main.loc[main['event']=='detail', :])

        _main = main.merge(sub_item, how='left', on='news_id')
        _main['event_news_type'] = _main['event']+'_'+_main['news_type']
        try:
            self.click_free_uid  = get_uid(_main.loc[_main['event_news_type']=='click_free', :])
        except:
            self.click_free_uid = None
        try:
            self.click_paid_uid  = get_uid(_main.loc[_main['event_news_type']=='click_paid', :])
        except:
            self.click_paid_uid  = None
        try:
            self.detail_free_uid = get_uid(_main.loc[_main['event_news_type']=='detail_free', :])
        except:
            self.detail_free_uid = None
        try:
            self.detail_paid_uid = get_uid(_main.loc[_main['event_news_type']=='detail_paid', :])
        except:
            self.detail_paid_uid = None

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
        
        if self.click_free_uid:
            click_free_info = self.click_free_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
            click_free_info = {'click_free_'+key:val for key,val in click_free_info.items()}
        else:
            click_free_info = {}
            
        if self.click_paid_uid:
            click_paid_info = self.click_paid_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
            click_paid_info = {'click_paid_'+key:val for key,val in click_paid_info.items()}
        else:
            click_paid_info = {}

        if self.detail_free_uid:
            detail_free_info = self.detail_free_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
            detail_free_info = {'detail_free_'+key:val for key,val in detail_free_info.items()}
        else:
            detail_free_info = {}

        if self.detail_paid_uid:
            detail_paid_info = self.detail_paid_uid.get_past_cnt_by_user_datetime(user_id, datetime, diff_days)
            detail_paid_info = {'detail_paid_'+key:val for key,val in detail_paid_info.items()}
        else:
            detail_paid_info = {}

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
    print('処理が重い前処理部分を並列処理する。')
    try:
        n_data_split = 2**8
        user_ids = np.unique(main.user_id)
        n_unit = int(user_ids.size / n_data_split)
        list_user_ids = [user_ids[i*n_unit:(i+1)*n_unit] for i in range(n_data_split)]
        list_of_key_args = [{'user_ids':list(user_ids), 'now_datetime':now_datetime, 'before_datetime':before_datetime} for user_ids in list_user_ids]
        df_list = multi(list_of_key_args)
        df = pd.concat(df_list, axis=0, ignore_index=True)
    except:
        import traceback
        traceback.print_exc()
        
    
    print('処理した結果をディスクに保存しておく')
    df.to_csv('output/prepreprocessed_data.csv', index=False)
    
    print('決定木の図などを出力する。')
    #df = pd.read_csv('output/prepreprocessed_data.csv')        
    main_process(df)
    print('finish')
