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
parser.add_argument('--lr',type=int,default=0.0001,help='lr')
parser.add_argument('--epochs',type=int,default=5,help='epochs')
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
            # loss = F.mse_loss(out.float(), temp_y.float())
            loss = F.l1_loss(out.float(), temp_y.float())
            result_df.iloc[idx+seq_len, :] = out.cpu().detach().numpy().item(), temp_y.cpu().detach().numpy().item()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            print(loss)
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
    dataset_1 = pd.read_csv(f'./dataset/solar_weather_2017_2020_남제주소내.csv', encoding='cp949', index_col=0)
    dataset_2 = pd.read_csv(f'./dataset/solar_weather_2017_2020_부산복합자재창고.csv', encoding='cp949', index_col=0)
    dataset_3 = pd.read_csv(f'./dataset/solar_weather_2017_2020_영월본부.csv', encoding='cp949', index_col=0)
    dataset_4 = pd.read_csv(f'./dataset/solar_weather_2017_2020_인천수산정수장.csv', encoding='cp949', index_col=0)
    dataset_5 = pd.read_csv(f'./dataset/solar_weather_2017_2020_하동보건소.csv', encoding='cp949', index_col=0)
    dataset_6 = pd.read_csv(f'./dataset/solar_weather_2017_2020_신안.csv', encoding='cp949', index_col=0)

    dataset_transform_0 =data_scale(dataset_0, args.norm)
    dataset_transform_1 =data_scale(dataset_1, args.norm)
    dataset_transform_2 =data_scale(dataset_2, args.norm)
    dataset_transform_3 =data_scale(dataset_3, args.norm)
    dataset_transform_4 =data_scale(dataset_4, args.norm)
    dataset_transform_5 =data_scale(dataset_5, args.norm)
    dataset_transform_6 =data_scale(dataset_6, args.norm)
    
    model_0, result_df_0, train_loss_graph_0 = local_model(dataset_transform_0, args)
    model_1, result_df_1, train_loss_graph_1 = local_model(dataset_transform_1, args)
    model_2, result_df_2, train_loss_graph_2 = local_model(dataset_transform_2, args)
    model_3, result_df_3, train_loss_graph_3 = local_model(dataset_transform_3, args)
    model_4, result_df_4, train_loss_graph_4 = local_model(dataset_transform_4, args)
    model_5, result_df_5, train_loss_graph_5 = local_model(dataset_transform_5, args)
    model_6, result_df_6, train_loss_graph_6 = local_model(dataset_transform_6, args)
    
    print(train_loss_graph_0)
    print(train_loss_graph_1)
    print(train_loss_graph_2)
    print(train_loss_graph_3)
    print(train_loss_graph_4)
    print(train_loss_graph_5)
    print(train_loss_graph_6)
    
    result_df_0.to_csv(f'{result_dir}/pred_result_0.csv')
    result_df_1.to_csv(f'{result_dir}/pred_result_1.csv')
    result_df_2.to_csv(f'{result_dir}/pred_result_2.csv')
    result_df_3.to_csv(f'{result_dir}/pred_result_3.csv')
    result_df_4.to_csv(f'{result_dir}/pred_result_4.csv')
    result_df_5.to_csv(f'{result_dir}/pred_result_5.csv')
    result_df_6.to_csv(f'{result_dir}/pred_result_6.csv')
    
    train_loss_graph_0.to_csv(f'{result_dir}/loss_graph_0.csv')
    train_loss_graph_1.to_csv(f'{result_dir}/loss_graph_1.csv')
    train_loss_graph_2.to_csv(f'{result_dir}/loss_graph_2.csv')
    train_loss_graph_3.to_csv(f'{result_dir}/loss_graph_3.csv')
    train_loss_graph_4.to_csv(f'{result_dir}/loss_graph_4.csv')
    train_loss_graph_5.to_csv(f'{result_dir}/loss_graph_5.csv')
    train_loss_graph_6.to_csv(f'{result_dir}/loss_graph_6.csv')
    
    torch.save(model_0, "./model_0.pt")
    torch.save(model_1, "./model_1.pt")
    torch.save(model_2, "./model_2.pt")
    torch.save(model_3, "./model_3.pt")
    torch.save(model_4, "./model_4.pt")
    torch.save(model_5, "./model_5.pt")
    torch.save(model_6, "./model_6.pt")
    
    # model_loaded = torch.load('./test_model.pt').to(device)
    
if __name__ == '__main__':
    main()