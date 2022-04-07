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
import gc

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda')
parser.add_argument('--data',type=str,default='busan_incheon_hadong_solarratio_temp_pre_pm_time2d_2013_2020.npy',help='data path')
parser.add_argument('--features', type=list, default=[0,1,2,3,4,5,6,7], help='feature index')   #input_size도 함께 변경
parser.add_argument('--input_size',type=int,default=15,help='input size')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--model', type=any, default=CNN1, help='model')
parser.add_argument('--hist_len',type=int,default=24*7,help='hist len')
parser.add_argument('--pred_len',type=int,default=1,help='pred len')
parser.add_argument('--hidden_size',type=int,default=512,help='hidden size')
parser.add_argument('--num_layers',type=int,default=8,help='num layers')
parser.add_argument('--norm', type=str, default='MinMax',help='Normalization Type')
parser.add_argument('--lr',type=int,default=0.001,help='lr')
parser.add_argument('--epochs',type=int,default=2,help='epochs')
parser.add_argument('--year_term',type=int,default=[2017010101,2020122509], help='start year ~ end year')   #feature nan값이 없는 최대 범위
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


def train(model, optimizer, train_batch, backprop, device):
    model.train()
    print(len(train_batch))
    train_loss = 0.0
    index_range = [r for r in arrow.Arrow.range('hour', arrow.get(str(args.year_term[0]), 'YYYYMMDDHH', tzinfo='Asia/Seoul'), arrow.get(str(args.year_term[1]), 'YYYYMMDDHH', tzinfo='Asia/Seoul'))]
    result_df = pd.DataFrame(columns=['model_prediction', 'real_value'], index=index_range)
    result_idx = args.hist_len
    cnt = 0
    for idx, data in enumerate(train_batch):
        print(cnt, ' ', end='')
        print(data.size())
        x, y = data[:, :args.hist_len, :].to(device), data[:,args.hist_len:,0].to(device)     #x : batch_size * seq_len * input_size
        print(x.size())
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out.float(), y.float())
        result_df.to_numpy()[result_idx:result_idx+args.batch_size, :] = np.vstack((out[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy())).T
        result_idx+=args.batch_size
        if backprop:
            loss.backward()
            optimizer.step()
        train_loss += loss.item()
        cnt+=1
    return
    size = len(train_batch.dataset)
    print(f"size : {size}")
    avg_loss = train_loss/size
    return avg_loss, result_df
        

def test(model, test_batch, device):
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    test_loss = 0.0
    index_range = [r for r in arrow.Arrow.range('hour', arrow.get(str(args.year_term[0]), 'YYYYMMDDHH', tzinfo='Asia/Seoul'), arrow.get(str(args.year_term[1]), 'YYYYMMDDHH', tzinfo='Asia/Seoul'))]
    result_df = pd.DataFrame(columns=['model_prediction', 'real_value'], index=index_range)
    result_idx = args.hist_len
    mape = MAPE()
    test_loss_mse = 0.0
    test_loss_mape = 0.0
    for idx, data in enumerate(test_batch):
        x, y = data[:, :args.hist_len, :].to(device), data[:,args.hist_len:,0].to(device)
        out = model(x)
        loss_mse = F.mse_loss(out.float(), y.float())
        loss_mape = mape.loss(y_pred = out.float(), target = y.float())
        # print(loss_mape)
        result_df.to_numpy()[result_idx:result_idx+args.batch_size, :] = np.vstack((out[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy())).T
        result_idx+=args.batch_size
        test_loss_mse += loss_mse.item()
        test_loss_mape += loss_mape
        del loss_mse, loss_mape
    size = len(test_batch.dataset)
    avg_loss_mse = test_loss_mse/size
    avg_loss_mape = test_loss_mape/size
    return avg_loss_mse, avg_loss_mape, result_df
    

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

    
    device = torch.device(args.device)
    
    
    location_list = config['experiments']['location_used'].copy()
    feature_list = config['experiments']['features_used'].copy()

    print(location_list)
    print(feature_list)

    for train_location in config['experiments']['location_used']:
        location_list = config['experiments']['location_used'].copy()
        location_list.remove(train_location)
        test_location_0 = location_list[0]
        test_location_1 = location_list[1]
        test_location_2 = location_list[2]
        test_location_3 = location_list[3]
        test_location_4 = location_list[4]
        test_location_5 = location_list[5]
        print(f"\nTrain Area : {train_location}\n")
        #   , test_location_0, test_location_1, test_location_2, test_location_3, test_location_4, test_location_5)
        
        train_dataset  = pd.read_csv(f'./dataset/solar_weather_2017_2020_{train_location}.csv', encoding='cp949')#, index_col=0)
        test_dataset_0 = pd.read_csv(f'./dataset/solar_weather_2017_2020_{test_location_0}.csv', encoding='cp949', index_col=0)
        test_dataset_1 = pd.read_csv(f'./dataset/solar_weather_2017_2020_{test_location_1}.csv', encoding='cp949', index_col=0)
        test_dataset_2 = pd.read_csv(f'./dataset/solar_weather_2017_2020_{test_location_2}.csv', encoding='cp949', index_col=0)
        test_dataset_3 = pd.read_csv(f'./dataset/solar_weather_2017_2020_{test_location_3}.csv', encoding='cp949', index_col=0)
        test_dataset_4 = pd.read_csv(f'./dataset/solar_weather_2017_2020_{test_location_4}.csv', encoding='cp949', index_col=0)
        test_dataset_5 = pd.read_csv(f'./dataset/solar_weather_2017_2020_{test_location_5}.csv', encoding='cp949', index_col=0)
        print(train_dataset.info())

        train_dataset = KORCSVDataset(data=train_dataset, locals=None, seq_len=args.hist_len+args.pred_len, norm='MinMax')
        # test_dataset_0 = KORCSVDataset(data=test_dataset_0, locals=None, seq_len=args.seq_len, norm='MinMax')
        # test_dataset_1 = KORCSVDataset(data=test_dataset_1, locals=None, seq_len=args.seq_len, norm='MinMax')
        # test_dataset_2 = KORCSVDataset(data=test_dataset_2, locals=None, seq_len=args.seq_len, norm='MinMax')
        # test_dataset_3 = KORwCSVDataset(data=test_dataset_3, locals=None, seq_len=args.seq_len, norm='MinMax')
        # test_dataset_4 = KORCSVDataset(data=test_dataset_4, locals=None, seq_len=args.seq_len, norm='MinMax')
        # test_dataset_5 = KORCSVDataset(data=test_dataset_5, locals=None, seq_len=args.seq_len, norm='MinMax')

        print(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
        # for idx, data in enumerate(train_loader):
        #     print(idx, data)
        #     print(data.shape)
        #     break
        print(len(train_loader))
        model = args.model(hist_len=args.hist_len, pred_len=args.pred_len, input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, device=args.device, args=args).to(args.device)
        loss_list = []
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print(args.device)
        
        for epoch in range(args.epochs):
            train_loss, result_train = train(model, optimizer, train_loader, backprop=args.backprop, device=args.device)
            
            loss_list.append(train_loss)
        plt.plot(loss_list)
        plt.show()
        plt.savefig(f"{result_dir}/train_loss.png")
        result_train.to_csv(f'{result_dir}/pred_result.csv')

        for test_location in [test_location_0, test_location_1, test_location_2, test_location_3, test_location_4, test_location_5]:
            print(f"\nTest Area == {test_location}\n")
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)
            avg_loss_mse, avg_loss_mape, result_test = test(model, test_loader, args.device)
            result_test.to_csv(f'{result_dir}/pred_result_{test_num}.csv')
            
            print("\n")
    
    
    
    
    
    
    # for train_dataset in [train_dataset_0]:     #여기서 학습
    #     print(f"\nTrain Area : {args.train_area}\n")
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)

    #     model = args.model(hist_len=args.hist_len, pred_len=args.pred_len, input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, device=args.device).to(args.device)
        
    #     loss_list = []
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    #     for epoch in range(args.epochs):
    #         train_loss, result_train = train(model, optimizer, train_loader, backprop=args.backprop, device=args.device)
            
    #         loss_list.append(train_loss)
    #     plt.plot(loss_list)
    #     plt.show()
    #     plt.savefig(f"{result_dir}/_train_loss.png")
    #     result_train.to_csv(f'{result_dir}/pred_result.csv')
    
    # test_num = 1
    # for test_dataset in [test_dataset_1, test_dataset_2]:   #여기서 예측
    #     print(f"\nTest Area\n")
    #     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)
    #     avg_loss_mse, avg_loss_mape, result_test = test(model, test_loader, args.device)
    #     result_test.to_csv(f'{result_dir}/pred_result_{test_num}.csv')
    #     test_num+=1
        
    #     print(f"avg_loss_mse : {avg_loss_mse}")
    #     print(f"avg_loss_mape : {avg_loss_mape}")
        
    
if __name__ == '__main__':
    main()
    