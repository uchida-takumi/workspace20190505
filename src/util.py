#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
便利関数
"""

import pandas as pd

def labeled_qcut(*pos_args, **key_args):
    '''
    EXAMPLE
    ----------------
    x = range(10)
    q = [0.0, 0.33, 0.66, 1.0]

    print( labeled_qcut(x, q) )
     > """
        ['000_(-0.001, 2.97]',
         '000_(-0.001, 2.97]',
         '000_(-0.001, 2.97]',
         '001_(2.97, 5.94]',
         '001_(2.97, 5.94]',
         '001_(2.97, 5.94]',
         '002_(5.94, 9.0]',
         '002_(5.94, 9.0]',
         '002_(5.94, 9.0]',
         '002_(5.94, 9.0]']
       """

    '''
    qcuted = pd.qcut(*pos_args, **key_args)
    if hasattr(qcuted, 'categories'):    
        labels = {val:'{}_{}'.format('%03d'%i, val) for i,val in enumerate(qcuted.categories)}
    else:
        labels = {val:'{}_{}'.format('%03d'%i, val) for i,val in enumerate(qcuted.dtype.categories)}

    result = [labels.get(qcut) for qcut in qcuted]
    return result

def labeled_cut(*pos_args, **key_args):
    '''
    EXAMPLE
    ----------------
    x = range(10)
    bins = [-0.1,3,8,10]

    print( labeled_cut(x, bins) )
     > """
        ['000_(-0.1, 3.0]',
         '000_(-0.1, 3.0]',
         '000_(-0.1, 3.0]',
         '000_(-0.1, 3.0]',
         '001_(3.0, 8.0]',
         '001_(3.0, 8.0]',
         '001_(3.0, 8.0]',
         '001_(3.0, 8.0]',
         '001_(3.0, 8.0]',
         '002_(8.0, 10.0]']
       """

    '''
    cuted = pd.cut(*pos_args, **key_args)
    if isinstance(cuted, pd.core.categorical.Categorical):
        labels = {val:'{}_{}'.format('%03d'%i, val) for i,val in enumerate(cuted.categories)}
    else:
        labels = {val:'{}_{}'.format('%03d'%i, val) for i,val in enumerate(cuted.dtype.categories)}
    result = [labels.get(cut) for cut in cuted]
    return result
