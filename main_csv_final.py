import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import arrow
import pandas as pd
from dataset import *
from utils import *
from models import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
# from pytorch_forecasting.metrics import MAPE
# import gc

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda')
# parser.add_argument('--data',type=str,default='busan_incheon_hadong_solarratio_temp_pre_pm_time2d_2013_2020.npy',help='data path')
# parser.add_argument('--features', type=list, default=[0,1,2,3,4,5,6,7], help='feature index')   #input_size도 함께 변경
parser.add_argument('--input_size',type=int,default=14,help='input size')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--model', type=any, default=DNN1, help='model')
parser.add_argument('--hist_len',type=int,default=24*3,help='hist len')
parser.add_argument('--pred_len',type=int,default=1,help='pred len')
parser.add_argument('--hidden_size',type=int,default=512,help='hidden size')
parser.add_argument('--num_layers',type=int,default=8,help='num layers')
parser.add_argument('--norm', type=str, default='Standard',help='Normalization Type')
parser.add_argument('--lr',type=int,default=0.001,help='lr')
parser.add_argument('--epochs',type=int,default=10,help='epochs')
parser.add_argument('--year_term',type=int,default=[2017010101,2021010101], help='start year ~ end year')   #feature nan값이 없는 최대 범위
parser.add_argument('--train_area', type=str,default='Busan', help='Train Area')        #train 외 지역은 test 지역
parser.add_argument('--backprop', type=bool, default=True, help='Backprop')
parser.add_argument('--metrics', type=str, default='l2', help='metrics')
# parser.add_argument('--test_area', type=str,default='Hadong', help='Test Area')

args = parser.parse_args()

# args.input_size = len(args.features)
# total_data = load_data(args.data)


# location_var = config['experiments']['location_used']
# weather_var = config['experiments']['features_used']
#* loc_list = [location_var.index('부산복합자재창고'), location_var.index('인천수산정수장'), location_var.index('하동보건소')] - 평균 유사한 것들
#* weather_list = [weather_var.index('발전률'), weather_var.index('기온(°C)'),weather_var.index('강수량(mm)'), weather_var.index('PM10'), weather_var.index('측정일시')]


#* main으로 넣어서 모두 가능하게 하기!




#! 여기까지는 얼추 완료
#! 여기까지는 얼추 완료
#! 여기까지는 얼추 완료


#* TO-DO
"""
발전률과의 비율 계산
계산된 비율 기반 weight matrix 생성
각 지역별로 모델 생성 후 학습



"""

#!data scaler
def data_scale(data, norm):
    if norm == "MinMax":
        scaler = MinMaxScaler()
    elif norm == "Standard":
        scaler = StandardScaler()
    else:
        NotImplementedError('wrong scaler type')
    data_transform = data.copy()
    data_transform[:] = scaler.fit_transform(data)
    return data_transform



def local_model(data, args):

    model = args.model(hist_len=args.hist_len, pred_len=args.pred_len, input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, device=args.device).to(args.device)
    model.train()
    train_loss_graph = pd.DataFrame(columns=['train_loss'], index=[i for i in range(args.epochs)])
    result_df = pd.DataFrame(columns=['model_prediction', 'real_value'], index=data.index)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    seq_len = args.hist_len + args.pred_len
    idx_range = len(data) - seq_len
    # X, y = data.iloc[:, 1:], data.iloc[:,0]
    X, y = data.iloc[:, 1:].to_numpy(), data.iloc[:,0].to_numpy()
    
    for epoch in range(args.epochs):
        print(epoch)
        train_loss = 0.0
        for idx in range(0, idx_range):
            # temp_X = torch.FloatTensor(X.iloc[idx:idx+args.hist_len, :].values).to(args.device)
            temp_X = torch.FloatTensor(X[idx:idx+args.hist_len, :]).to(args.device)
            temp_y = torch.FloatTensor([y[idx+seq_len]]).to(args.device)
            optimizer.zero_grad()
            out =model(temp_X)
            loss = F.mse_loss(out.float(), temp_y.float())
            result_df.iloc[idx+seq_len, :] = out.cpu().detach().numpy().item(), temp_y.cpu().detach().numpy().item()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        train_loss_graph.loc[epoch, 'train_loss'] = train_loss
    
    return model, result_df, train_loss_graph



def main():
    
    start_time = arrow.now().format('YYYYMMDDHHmmss')
    print(start_time)
    
    device = torch.device(args.device)
    print(device)
    
    try:
        os.mkdir(os.getcwd() + f"/result")
    except:
        pass
    os.mkdir(os.getcwd() + f"/result/{start_time}_hist{args.hist_len}_pred{args.pred_len}_model{args.model.__name__}_epoch{args.epochs}_train{args.train_area}")
    result_dir = f"{os.getcwd()}/result/{start_time}_hist{args.hist_len}_pred{args.pred_len}_model{args.model.__name__}_epoch{args.epochs}_train{args.train_area}"
    print(f"Directory Created")

    
    
    dataset_0 = pd.read_csv(f'./dataset/solar_weather_2017_2020_경상대.csv', encoding='cp949', index_col=0)
    dataset_transform_0 =data_scale(dataset_0, args.norm)
    model_0, result_df_0, train_loss_graph_0 = local_model(dataset_transform_0, args)
    print(train_loss_graph_0)
    
    result_df_0.to_csv(f'{result_dir}/pred_result_0.csv')
    train_loss_graph_0.to_csv(f'{result_dir}/loss_graph_0.csv')
    
    
if __name__ == '__main__':
    main()