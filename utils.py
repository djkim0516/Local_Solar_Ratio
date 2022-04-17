import numpy as np
import dtw
import os
import yaml
import arrow
from collections import Counter 


proj_dir = os.getcwd()
conf_fp = os.path.join(proj_dir, 'config.yaml')
conf_original = os.path.join(proj_dir, 'config copy.yaml')
with open(conf_fp, encoding='UTF8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open(conf_original, encoding='UTF8') as f:
    config_copy = yaml.load(f, Loader=yaml.FullLoader)


# ffill along axis 1, as provided in the answer by Divakar
def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out


# Simple solution for bfill provided by financial_physician in comment below
def bfill(arr): 
    return ffill(arr[:, ::-1])[:, ::-1]

def get_nonan_mean(arr):
    return np.mean(arr[~np.isnan(arr.astype(float))]).round(1)

def dtw_distance(array1, array2, seq_start, seq_end, idx):
    '''
    1차원 벡터 두 개, 기간, feature index 받아서
    ffill, bfill로 nan값 처리 후 dtw 값 반환
    '''
    arr1 = array1[seq_start:seq_end, idx].astype(float)
    arr2 = array2[seq_start:seq_end, idx].astype(float)
    if idx==1:
        arr1 = np.nan_to_num(arr1, nan=0.0)
        arr2 = np.nan_to_num(arr2, nan=0.0)
    else:
        arr1 = bfill(ffill(np.expand_dims(arr1, axis=0)))
        arr2 = bfill(ffill(np.expand_dims(arr2, axis=0)))
        
    return dtw.dtw(arr1, arr2, keep_internals=True)
    
def get_idx(t):
    t0=arrow.get("2013010101", 'YYYYMMDDHH', tzinfo='Asia/Seoul')   #varies by file
    t=arrow.get(str(t), 'YYYYMMDDHH', tzinfo='Asia/Seoul')
    return int((t.timestamp() - t0.timestamp()) / (60 * 60))    #Return Hourly Data Index

def convert_2d(t):
    
    idx = get_idx(t)
    year = arrow.get(str(t), 'YYYYMMDDHH', tzinfo='Asia/Seoul').date().year
    month = arrow.get(str(t), 'YYYYMMDDHH', tzinfo='Asia/Seoul').date().month
    day = arrow.get(str(t), 'YYYYMMDDHH', tzinfo='Asia/Seoul').date().day
    
    if year >= 2016:
        if month >= 3:
            idx -= 24
            
    if year >= 2020:
        if month >= 3:
            idx -= 24    
            
    date = idx // 24 % 365
    hour = idx % 24 + 1
    
    date_cos = np.cos(date * 2 * np.pi / 365)
    date_sin = np.sin(date * 2 * np.pi / 365)
    
    hour_cos = np.cos(hour * 2 * np.pi / 24)
    hour_sin = np.sin(hour * 2 * np.pi / 24)
    
    return date_cos, date_sin, hour_cos, hour_sin


def mape_loss(y_pred, target):
    # print((y_pred - target).abs())
    loss = (y_pred - target).abs() / (target.abs() + 1e-4)
    return loss


def detect_outliers(df, n, features): 
    outlier_indices = [] 
    for col in features: 
        
        Q1 = np.percentile(df[col], 25) 
        Q3 = np.percentile(df[col], 75) 
        IQR = Q3 - Q1 
        outlier_step = 1.5 * IQR 
        print(f"Counting Outliers in {col}...")
        print(Q1 - outlier_step, Q3 + outlier_step)
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index 
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices) 
    # print(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n) 

    # return multiple_outliers
    return df[~df.index.isin(multiple_outliers)]