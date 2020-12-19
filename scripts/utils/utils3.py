#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
# Complement Nan with previous data 
def nan_pre_data(df,ffill=[1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18]):
    #上記のポイントはひとつ前の値で補完する
    df_tmp = df.copy()
    df_tmp = df_tmp.replace(0.0, np.nan)
    df_tmp = df_tmp.fillna(method='ffill', limit=10)
    df_tmp = df_tmp.replace(np.nan,0.0)

    return df_tmp

def std(df):
    df = (df - df.min())/(df.max() - df.min())
    return df

def make_std(df):
    df_tmp = df.copy()
    df_train = df_tmp.apply(std,axis=1)
    df_train["location_x"] = df["x_Neck"]/640
    df_train["location_y"] = df["y_Neck"]/480
    df_train["size_y"] = (df["y_Hip"]-df["y_Neck"])/480
    return df_train


