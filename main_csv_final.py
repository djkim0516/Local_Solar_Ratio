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


