from threading import local
import numpy as np
from utils import *
import torch
from torch.utils.data import Dataset
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class KORDataset(Dataset):
    
    def __init__(self, seq_len, locals=[0,1,2], features=[0,1,2,3,4,5,6,7], year=[2013010101,2021010101], norm='Standard', data_path='busan_incheon_hadong_solarratio_temp_pre_pm_time2d_2013_2020.npy', **kwargs):
        # super(KORDataset, self).__init__()
        
        self.locals = locals
        self.features = features
        self.data_path = data_path
        self.seq_len = seq_len
        self.norm = norm 
        # self.hour_start = (year[0] - 2013) * 24 * 365
        # self.hour_end = (year[1] - 2013 + 1) * 24 * 365
        self.hour_start = get_idx(year[0])
        self.hour_end = get_idx(year[1])
        
        self.data = np.load(self.data_path, allow_pickle=True)
        self.data = self.data[self.locals][:,self.hour_start:self.hour_end, self.features].astype('float32')
        self.data[:,:,2] = np.nan_to_num(self.data[:,:,2], nan=0.0)     #강수량 nan -> 0
        self.data = np.nan_to_num(self.data.astype(float), nan=0.0)     #test************
        self.data = torch.from_numpy(self.data)

        if len(kwargs) == 0:
            self.scaler = self.norm_scaler(self.data, norm)     #scaler initialize
        else:
            for key, value in kwargs.items():
                if key == 'scaler':
                    scaler = value
                    self.data = np.expand_dims(scaler.transform(np.squeeze(self.data, axis=0)), axis=0)       #data normalize
            
                
        
    def norm_scaler(self, data, norm):
        if norm == 'Standard':
            scaler = StandardScaler()
        elif norm == 'MinMax':
            scaler = MinMaxScaler()
        # print(data.shape)
        self.data = np.expand_dims(scaler.fit_transform(np.squeeze(data, axis=0)), axis=0)
        # self.data = scaler.transform(data)
        return scaler
        
    def __len__(self):
        
        # print(self.data.shape)
        # if len(self.locals) == 1:
        #     return self.data.shape[1] - (self.seq_len - 1)
        # else :
        return self.data.shape[1] - (self.seq_len - 1)

    # def __getitem__(self, local_index, hour_index):
    #     if isinstance(local_index, slice):
    #         return self.data[[i for i in range(*local_index.indices)], hour_index:hour_index + self.seq_len, :]
    #     else:
    #         return super().__getitem__(local_index, hour_index)
    
    def __getitem__(self, index):
        # print(index)
        if len(self.locals) == 1:
            hour_index = index
            return self.data[0, hour_index:hour_index + self.seq_len, :]
        
        else:
            local_index, hour_index = index
            if isinstance(local_index, slice):
                return self.data[[i for i in range(*local_index.indices(local_index.stop))], hour_index:hour_index + self.seq_len, :]
            else:
                return self.data[local_index, hour_index:hour_index + self.seq_len, :].float()
    
    '''
    def __getitem__(self, index):
        
        return self.data[:][:, index]
    ''' 
    '''
    def _load_data(path):
        data = np.load(path, allow_pickle=True)
        return torch.from_numpy(data.values)
    '''


    # def _    
    '''
    하나씩 옮겨가는 dataset 생성
    '''
# class KORtimeDataset(TimeSeriesDataSet):
    