from tracemalloc import start
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import glob
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
parser.add_argument('--epochs',type=int,default=30,help='epochs')
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
    elif norm == "Robust":
        scaler = RobustScaler()
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
        print(f"Train Epoch : {epoch}")
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
            # print(loss)
        train_loss_graph.loc[epoch, 'train_loss'] = train_loss
    
    return model, result_df, train_loss_graph


# def test(data, model, args):
#     result =
#     return  result
def test(data, model, args):

    model.eval()
    test_loss_graph = pd.DataFrame(columns=['test_loss'], index=[i for i in range(args.epochs)])
    result_df = pd.DataFrame(columns=['model_prediction', 'real_value'], index=data.index)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    seq_len = args.hist_len + args.pred_len
    idx_range = len(data) - seq_len
    # X, y = data.iloc[:, 1:], data.iloc[:,0]
    X, y = data.iloc[:, 1:].to_numpy(), data.iloc[:,0].to_numpy()
    
    for epoch in range(1):
        print(epoch)
        test_loss = 0.0
        for idx in range(0, idx_range):
            # temp_X = torch.FloatTensor(X.iloc[idx:idx+args.hist_len, :].values).to(args.device)
            temp_X = torch.FloatTensor(X[idx:idx+args.hist_len, :]).to(args.device)
            temp_y = torch.FloatTensor([y[idx+seq_len]]).to(args.device)
            # optimizer.zero_grad()
            out =model(temp_X)
            # loss = F.mse_loss(out.float(), temp_y.float())
            loss = F.l1_loss(out.float(), temp_y.float())
            result_df.iloc[idx+seq_len, :] = out.cpu().detach().numpy().item(), temp_y.cpu().detach().numpy().item()
            # loss.backward()
            # optimizer.step()
            test_loss+=loss.item()
        # print(f"test loss : {test_loss/idx_range}")
        test_loss_graph.loc[epoch, 'test_loss'] = test_loss
    
    return result_df, test_loss_graph







def main():
    
    start_time = arrow.now()#.format('YYYYMMDDHHmmss')
    start_time_str = start_time.format('YYYYMMDDHHmmss')
    print(start_time)
    
    device = torch.device(args.device)
    print(device)
    
    try:
        os.mkdir(os.getcwd() + f"/result")
    except:
        pass
    os.mkdir(os.getcwd() + f"/result/{start_time_str}_hist{args.hist_len}_pred{args.pred_len}_model{args.model.__name__}_epoch{args.epochs}_train{args.train_area}")
    result_dir = f"{os.getcwd()}/result/{start_time_str}_hist{args.hist_len}_pred{args.pred_len}_model{args.model.__name__}_epoch{args.epochs}_train{args.train_area}"
    print(f"Directory Created")

    loc_list = ['경상대', '남제주소내', '부산복합자재창고', '영월본부', '인천수산정수장', '하동보건소', '신안']
    feature_list = ['기온(°C)','강수량(mm)','풍속(m__s)','습도(%)','증기압(hPa)','현지기압(hPa)','일조(hr)','지면온도(°C)','SO2','CO','O3','NO2','PM10','PM25']
    
    #! 데이터 불러온 후 정규화
    dataset_0 = pd.read_csv(f'./dataset/solar_weather_2017_2020_경상대.csv', encoding='cp949', index_col=0)
    dataset_1 = pd.read_csv(f'./dataset/solar_weather_2017_2020_남제주소내.csv', encoding='cp949', index_col=0)
    dataset_2 = pd.read_csv(f'./dataset/solar_weather_2017_2020_부산복합자재창고.csv', encoding='cp949', index_col=0)
    dataset_3 = pd.read_csv(f'./dataset/solar_weather_2017_2020_영월본부.csv', encoding='cp949', index_col=0)
    dataset_4 = pd.read_csv(f'./dataset/solar_weather_2017_2020_인천수산정수장.csv', encoding='cp949', index_col=0)
    dataset_5 = pd.read_csv(f'./dataset/solar_weather_2017_2020_하동보건소.csv', encoding='cp949', index_col=0)
    dataset_6 = pd.read_csv(f'./dataset/solar_weather_2017_2020_신안.csv', encoding='cp949', index_col=0)

    dataset = pd.concat([dataset_0, dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6], axis=0)

    dataset = data_scale(dataset, args.norm)    #한번에 정규화
    
    
    # dataset_transform_0 =data_scale(dataset_0, args.norm)
    # dataset_transform_1 =data_scale(dataset_1, args.norm)
    # dataset_transform_2 =data_scale(dataset_2, args.norm)
    # dataset_transform_3 =data_scale(dataset_3, args.norm)
    # dataset_transform_4 =data_scale(dataset_4, args.norm)
    # dataset_transform_5 =data_scale(dataset_5, args.norm)
    # dataset_transform_6 =data_scale(dataset_6, args.norm)
    
    dataset_transform_0 = dataset.iloc[0*35064:1*35064]
    dataset_transform_1 = dataset.iloc[1*35064:2*35064]
    dataset_transform_2 = dataset.iloc[2*35064:3*35064]
    dataset_transform_3 = dataset.iloc[3*35064:4*35064]
    dataset_transform_4 = dataset.iloc[4*35064:5*35064]
    dataset_transform_5 = dataset.iloc[5*35064:6*35064]
    dataset_transform_6 = dataset.iloc[6*35064:7*35064]
    
    dataset_transform = [dataset_transform_0, dataset_transform_1, dataset_transform_2, dataset_transform_3, dataset_transform_4, dataset_transform_5, dataset_transform_6] 
    
    #! 상관계수 계산하기 위해 낮1시 데이터만 취합
    dataset_sunny_0 = dataset_transform_0.iloc[12::24].copy()
    dataset_sunny_1 = dataset_transform_1.iloc[12::24].copy()
    dataset_sunny_2 = dataset_transform_2.iloc[12::24].copy()
    dataset_sunny_3 = dataset_transform_3.iloc[12::24].copy()
    dataset_sunny_4 = dataset_transform_4.iloc[12::24].copy()
    dataset_sunny_5 = dataset_transform_5.iloc[12::24].copy()
    dataset_sunny_6 = dataset_transform_6.iloc[12::24].copy()
    
    #!이상치 제거
    dataset_sunny_0 = detect_outliers(dataset_sunny_0, 1, dataset_sunny_0.columns)
    dataset_sunny_1 = detect_outliers(dataset_sunny_1, 1, dataset_sunny_1.columns)
    dataset_sunny_2 = detect_outliers(dataset_sunny_2, 1, dataset_sunny_2.columns)
    dataset_sunny_3 = detect_outliers(dataset_sunny_3, 1, dataset_sunny_3.columns)
    dataset_sunny_4 = detect_outliers(dataset_sunny_4, 1, dataset_sunny_4.columns)
    dataset_sunny_5 = detect_outliers(dataset_sunny_5, 1, dataset_sunny_5.columns)
    dataset_sunny_6 = detect_outliers(dataset_sunny_6, 1, dataset_sunny_6.columns)
    
    dataset_corr_0 = dataset_sunny_0.corr().loc['발전률'][1:]
    dataset_corr_1 = dataset_sunny_1.corr().loc['발전률'][1:]
    dataset_corr_2 = dataset_sunny_2.corr().loc['발전률'][1:]
    dataset_corr_3 = dataset_sunny_3.corr().loc['발전률'][1:]
    dataset_corr_4 = dataset_sunny_4.corr().loc['발전률'][1:]
    dataset_corr_5 = dataset_sunny_5.corr().loc['발전률'][1:]
    dataset_corr_6 = dataset_sunny_6.corr().loc['발전률'][1:]
    # print(dataset_corr_0)
    # print(dataset_corr_1)
    # print(dataset_corr_2)
    # print(dataset_corr_3)
    # print(dataset_corr_4)
    # print(dataset_corr_5)
    # print(dataset_corr_6)
    
    solar_to_feature_corr = pd.concat([dataset_corr_0, dataset_corr_1, dataset_corr_2, dataset_corr_3, dataset_corr_4, dataset_corr_5, dataset_corr_6], axis=1)
    solar_to_feature_corr.columns = loc_list
    solar_to_feature_corr.to_csv(f'{result_dir}/solar_to_feature_corr.csv', encoding='cp949')      #*지역별 발전률-변수 corr
    # print(solar_to_feature_corr.shape)
    
        
    
    
    #! 모델 존재 여부 확인 후 학습 or 불러오기
    
    if len(glob.glob('./models/*.pt')) == 7:
        # model_loaded = torch.load('./test_model.pt').to(device)
        # loc_list = ['경상대', '남제주소내', '부산복합자재창고', '영월본부', '인천수산정수장', '하동보건소', '신안']
        
        model_0 = torch.load(glob.glob('./models/*model_0.pt')[0]).to(device)       #경상대
        model_1 = torch.load(glob.glob('./models/*model_1.pt')[0]).to(device)       #남제주소내
        model_2 = torch.load(glob.glob('./models/*model_2.pt')[0]).to(device)       #부산복합자재창고
        model_3 = torch.load(glob.glob('./models/*model_3.pt')[0]).to(device)       #영월본부
        model_4 = torch.load(glob.glob('./models/*model_4.pt')[0]).to(device)       #인천수산정수장
        model_5 = torch.load(glob.glob('./models/*model_5.pt')[0]).to(device)       #하동보건소
        model_6 = torch.load(glob.glob('./models/*model_6.pt')[0]).to(device)       #신안
        
    
    else:
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
        
        torch.save(model_0, f"./models/{start_time_str}_model_0.pt")
        torch.save(model_1, f"./models/{start_time_str}_model_1.pt")
        torch.save(model_2, f"./models/{start_time_str}_model_2.pt")
        torch.save(model_3, f"./models/{start_time_str}_model_3.pt")
        torch.save(model_4, f"./models/{start_time_str}_model_4.pt")
        torch.save(model_5, f"./models/{start_time_str}_model_5.pt")
        torch.save(model_6, f"./models/{start_time_str}_model_6.pt")
    
    # model_loaded = torch.load('./test_model.pt').to(device)
    
    #! 학습된 모델로 지역별로 돌아가며 예측
    
    model_list = [model_0, model_1, model_2, model_3, model_4, model_5, model_6]
    
    model_list = [model_0, model_1, model_2, model_3, model_4, model_5, model_6]
    num_list = [num for num in range(len(model_list))]
    for target_num in num_list:     #target_num 예측 위치
        print(f"Target Area : {target_num}")
        num_list_copy = num_list[:]
        target_model = model_list[num_list_copy.pop(target_num)]
        # num_list_copy 는 나머지 6개 지역
        train_model_0, train_model_1, train_model_2, train_model_3, train_model_4, train_model_5 = \
        model_list[num_list_copy[0]], model_list[num_list_copy[1]], model_list[num_list_copy[2]], model_list[num_list_copy[3]], model_list[num_list_copy[4]], model_list[num_list_copy[5]]
        
        
        local_feature_corr = pd.DataFrame()
        for feature in feature_list:
            feature_corr = pd.read_csv(f"./dataset/corr/feature_corr_2017_2020_{feature}.csv", encoding='cp949', index_col=0)
            local_feature_corr = pd.concat([local_feature_corr, feature_corr.iloc[num_list_copy, target_num]], axis=1)
        # print(solar_to_feature_corr.shape)
        # print(local_feature_corr.shape)
        local_feature_corr.columns = feature_list
        local_feature_corr.to_csv(f"./dataset/corr/local_feature_corr_{target_num}.csv", encoding='cp949')

        local_solar_to_feature_corr = solar_to_feature_corr.iloc[:, num_list_copy].T.copy()    #* target 뺀 나머지
        # print(local_solar_to_feature_corr.shape)
        local_weight = pd.DataFrame(np.zeros((1,6)))
        for loc_num in local_weight.columns:
            local_weight[loc_num] = np.matmul(abs(local_solar_to_feature_corr.iloc[loc_num].to_numpy()), abs(local_feature_corr.iloc[loc_num].to_numpy()))
        
        local_weight = local_weight.div(local_weight.sum(axis=1), axis=0)   #* 계산된 비율
        local_weight.to_csv(f"{result_dir}/local_weight_{target_num}.csv")
        print(local_weight)
        target_data = dataset_transform[target_num]#.iloc[:, 1:]      #* 
        # print(target_data)
        
        test_result_df_0, test_loss_0 = test(target_data, train_model_0, args)
        test_result_df_1, test_loss_1 = test(target_data, train_model_1, args)
        test_result_df_2, test_loss_2 = test(target_data, train_model_2, args)
        test_result_df_3, test_loss_3 = test(target_data, train_model_3, args)
        test_result_df_4, test_loss_4 = test(target_data, train_model_4, args)
        test_result_df_5, test_loss_5 = test(target_data, train_model_5, args)
        
        pred_result = pd.concat([test_result_df_0.iloc[:,0], test_result_df_1.iloc[:,0], test_result_df_2.iloc[:,0], test_result_df_3.iloc[:,0], test_result_df_4.iloc[:,0], test_result_df_5], axis=1)
        pred_result['final_prediction'] = np.matmul(pred_result.iloc[:,:6], local_weight.T)
        print(pred_result)
        pred_result.to_csv(f"{result_dir}/pred_result_{target_num}.csv")
        
    
    end_time = arrow.now()
    
    print(f"Total elapsed time : {end_time - start_time}")
        
        
    
    
if __name__ == '__main__':
    main()