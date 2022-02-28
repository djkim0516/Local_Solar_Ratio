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
from pytorch_forecasting.metrics import MAPE
import gc

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda')
parser.add_argument('--data',type=str,default='busan_incheon_hadong_solarratio_temp_pre_pm_time2d_2013_2020.npy',help='data path')
parser.add_argument('--features', type=list, default=[0,1,2,3,4,5,6,7], help='feature index')   #input_size도 함께 변경
parser.add_argument('--input_size',type=int,default=8,help='input size')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--model', type=any, default=DNN2, help='model')
parser.add_argument('--hist_len',type=int,default=24*5,help='hist len')
parser.add_argument('--pred_len',type=int,default=1,help='pred len')
parser.add_argument('--hidden_size',type=int,default=256,help='hidden size')         #작으면 underfitting
parser.add_argument('--num_layers',type=int,default=2,help='nu  yers')
parser.add_argument('--norm', type=str, default='MinMax',help='Normalization Type')
parser.add_argument('--lr',type=int,default=0.001,help='lr')
parser.add_argument('--epochs',type=int,default=20,help='epochs')
parser.add_argument('--year_term',type=int,default=[2014010101,2021010101], help='start year ~ end year')   #feature nan값이 없는 최대 범위
parser.add_argument('--train_area', type=str,default='Busan', help='Train Area')        #train 외 지역은 test 지역
parser.add_argument('--backprop', type=bool, default=True, help='Backprop')
parser.add_argument('--metrics', type=str, default='l2', help='metrics')
# parser.add_argument('--test_area', type=str,default='Hadong', help='Test Area')

args = parser.parse_args()

args.input_size = len(args.features)
# total_data = load_data(args.data)


# location_var = config['experiments']['location_used']
# weather_var = config['experiments']['features_used']
#* loc_list = [location_var.index('부산복합자재창고'), location_var.index('인천수산정수장'), location_var.index('하동보건소')] - 평균 유사한 것들
#* weather_list = [weather_var.index('발전률'), weather_var.index('기온(°C)'),weather_var.index('강수량(mm)'), weather_var.index('PM10'), weather_var.index('측정일시')]


#* main으로 넣어서 모두 가능하게 하기!
if args.train_area == 'Busan':
    train_num = 0
    test_num_1 = 1
    test_num_2 = 2
elif args.train_area == 'Incheon':
    train_num = 1
    test_num_1 = 0
    test_num_2 = 2
elif args.train_area == 'Hadong':
    train_num = 2
    test_num_1 = 0
    test_num_2 = 1
else:
    NotImplementedError('Train Area not specified')
    
#dataset 받아오기
##메모리 부족 문제 가능성, 발생시 변수 선언 순서 변경
# train_dataset_0 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[train_num], features=args.features, year=args.year_term, norm=args.norm)
# scaler = train_dataset_0.scaler
# test_dataset_1 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[test_num_1], features=args.features, year=args.year_term, scaler=scaler)
# test_dataset_2 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[test_num_2], features=args.features, year=args.year_term, scaler=scaler)
# args.model = MultiChannelLSTM_DifferentTimeScale
if args.model == MultiChannelLSTM_DifferentTimeScale:
    train_dataset_00 = KORDataset(seq_len=6+args.pred_len, locals=[train_num], features=args.features, year=args.year_term, norm=args.norm)
    train_dataset_01 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[train_num], features=args.features, year=args.year_term, norm=args.norm)
    scaler = train_dataset_00.scaler
    test_dataset_10 = KORDataset(seq_len=6+args.pred_len, locals=[test_num_1], features=args.features, year=args.year_term, scaler=scaler)
    test_dataset_11 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[test_num_1], features=args.features, year=args.year_term, scaler=scaler)
    test_dataset_20 = KORDataset(seq_len=6+args.pred_len, locals=[test_num_2], features=args.features, year=args.year_term, scaler=scaler)
    test_dataset_21 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[test_num_2], features=args.features, year=args.year_term, scaler=scaler)
else:
    train_dataset_0 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[train_num], features=args.features, year=args.year_term, norm=args.norm)
    train_scaler = train_dataset_0.scaler
    test_dataset_1 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[test_num_1], features=args.features, year=args.year_term, norm=args.norm)
    test_scaler_1 = test_dataset_1.scaler
    test_dataset_2 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[test_num_2], features=args.features, year=args.year_term, norm=args.norm)
    test_scaler_2 = test_dataset_2.scaler



def train(model, optimizer, train_batch, backprop, device):
    model.train()
    print(len(train_batch))
    train_loss = 0.0
    index_range = [r for r in arrow.Arrow.range('hour', arrow.get(str(args.year_term[0]), 'YYYYMMDDHH', tzinfo='Asia/Seoul'), arrow.get(str(args.year_term[1]), 'YYYYMMDDHH', tzinfo='Asia/Seoul'))]
    result_df = pd.DataFrame(columns=['model_prediction', 'real_value'], index=index_range)
    result_idx = args.hist_len
    for idx, data in enumerate(train_batch):
        x, y = data[:, :args.hist_len, :].to(device), data[:,args.hist_len:,0].to(device)     #x : batch_size * seq_len * input_size
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out.float(), y.float())
        result_df.to_numpy()[result_idx:result_idx+args.batch_size, :] = np.vstack((out[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy())).T
        result_idx+=args.batch_size
        if backprop:
            loss.backward()
            optimizer.step()
        train_loss += loss.item()
    size = len(train_batch.dataset)
    print(f"size : {size}")
    avg_loss = train_loss/size
    return avg_loss, result_df
        

def test(model, test_batch, device):
    model.eval()
    test_loss = 0.0
    index_range = [r for r in arrow.Arrow.range('hour', arrow.get(str(args.year_term[0]), 'YYYYMMDDHH', tzinfo='Asia/Seoul'), arrow.get(str(args.year_term[1]), 'YYYYMMDDHH', tzinfo='Asia/Seoul'))]
    result_df = pd.DataFrame(columns=['model_prediction', 'real_value'], index=index_range)
    result_idx = args.hist_len
    # mape = MAPE()#
    test_loss_mse = 0.0
    test_loss_mape = 0.0
    for idx, data in enumerate(test_batch):
        gc.collect()
        torch.cuda.empty_cache()
        x, y = data[:, :args.hist_len, :].to(device), data[:,args.hist_len:,0].to(device)
        out = model(x)
        loss_mse = F.mse_loss(out.float(), y.float())
        # print(loss_mape)
        result_df.to_numpy()[result_idx:result_idx+args.batch_size, :] = np.vstack((out[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy())).T
        result_idx+=args.batch_size
        test_loss_mse += loss_mse.item()
        # print(torch.mean(mape_loss(y_pred = out.cpu().detach().float(), target = y.cpu().detach().float()), axis=0))
        test_loss_mape += torch.mean(mape_loss(y_pred = out.cpu().detach().float(), target = y.cpu().detach().float()), axis=0)##
        # del loss_mse, loss_mape
    size = len(test_batch.dataset)
    avg_loss_mse = test_loss_mse/size
    avg_loss_mape = test_loss_mape/size
    return avg_loss_mse, avg_loss_mape, result_df

   
def mape_loss(y_pred, target):
        loss = (y_pred - target).abs() / (target.abs() + 1e-4)
        return loss


def main():
    
    start_time = arrow.now().format('YYYYMMDDHHmmss')
    print(start_time)
    
    try:
        os.mkdir(os.getcwd() + f"/result")
    except:
        pass
    os.mkdir(os.getcwd() + f"/result/{start_time}_hist{args.hist_len}_pred{args.pred_len}_model{args.model.__name__}_epoch{args.epochs}_train{args.train_area}")
    result_dir = f"{os.getcwd()}/result/{start_time}_hist{args.hist_len}_pred{args.pred_len}_model{args.model.__name__}_epoch{args.epochs}_train{args.train_area}"
    print(f"Directory Created")

    loss_result = pd.DataFrame(index=[i for i in range(args.epochs + 6)], columns=['Busan', 'Incheon', 'Hadong']) #*각 지역별로 epoch 훈련 돌린거 + 각 지역에 대한 test loss 저장
    
    
    for args.train_area in ['Busan', 'Incheon', 'Hadong']:
        if args.train_area == 'Busan':
            train_num = 0
            test_num_1 = 1
            test_num_2 = 2
        elif args.train_area == 'Incheon':
            train_num = 1
            test_num_1 = 0
            test_num_2 = 2
        elif args.train_area == 'Hadong':
            train_num = 2
            test_num_1 = 0
            test_num_2 = 1
        else:
            NotImplementedError('Train Area not specified')
            
        train_dataset_0 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[train_num], features=args.features, year=args.year_term, norm=args.norm)
        train_scaler = train_dataset_0.scaler
        test_dataset_1 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[test_num_1], features=args.features, year=args.year_term, norm=args.norm)
        test_scaler_1 = test_dataset_1.scaler
        test_dataset_2 = KORDataset(seq_len=args.hist_len+args.pred_len, locals=[test_num_2], features=args.features, year=args.year_term, norm=args.norm)
        test_scaler_2 = test_dataset_2.scaler


    
        device = torch.device(args.device)
        
        
        for train_dataset in [train_dataset_0]:     #여기서 학습
            print(f"\nTrain Area : {args.train_area}\n")
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)

            model = args.model(hist_len=args.hist_len, pred_len=args.pred_len, input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, device=args.device).to(args.device)
            
            loss_list = []
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            
            for epoch in range(args.epochs):
                train_loss, result_train = train(model, optimizer, train_loader, backprop=args.backprop, device=args.device)
                
                loss_list.append(train_loss)
                loss_result.iloc[epoch, train_num] = train_loss
                
            # plt.plot(loss_list)
            # plt.show()
            # plt.savefig(f"{result_dir}/_train_loss.png")
            result_train.to_csv(f'{result_dir}/pred_result_train_{args.train_area}.csv')
        
        test_num = 1
        for test_dataset in [test_dataset_1, test_dataset_2]:   #여기서 예측
            print(f"\nTest Area\n")
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)
            avg_loss_mse, avg_loss_mape, result_test = test(model, test_loader, args.device)
            result_test.to_csv(f'{result_dir}/pred_result_train_{args.train_area}_testarea_{test_num}.csv')
            loss_result.iloc[epoch + test_num + 1, train_num] = avg_loss_mse
            loss_result.iloc[epoch + test_num + 4, train_num] = avg_loss_mape
            test_num+=1
        
            print(f"avg_loss_mse : {avg_loss_mse}")
            print(f"avg_loss_mape : {avg_loss_mape}")
        
    loss_result.to_csv(f'{result_dir}/loss_result.csv')
    
if __name__ == '__main__':
    main()
    