from audioop import bias
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class LSTM1(nn.Module):
    
    '''
    feature 개수 : input_size만큼 반영
    길이 : pred_len만큼 반영
    
    '''
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers):
        super(LSTM1, self).__init__()
        self.hist_len = hist_len        #
        self.pred_len = pred_len        #과거 길이, 예측 길이 나눠야됨
        self.num_layers = num_layers
        self.input_size = input_size    
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True)
        self.fc1 = nn.Linear(self.hidden_size, self.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.float()
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True)       #차원 확인
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True)
        print(h_0.shape)
        print(x.shape)
        #Propagate input through LSTM
        x = torch.tensor(x, dtype = torch.float32)
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc1(h_out)
        out = self.relu(out)
        
        return out

        
class LSTM2(nn.Module):
    
    '''
    feature 개수 : input_size만큼 반영
    길이 : pred_len만큼 반영
    
    '''
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers, device):
        super(LSTM2, self).__init__()
        self.hist_len = hist_len        #
        self.pred_len = pred_len        #과거 길이, 예측 길이 나눠야됨
        # self.num_features = num_features
        self.num_layers = num_layers
        self.input_size = input_size    
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True, dropout=0.3)
        self.fc1 = nn.Linear(self.hidden_size * self.num_layers, self.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.float()
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(self.device))       #차원 확인,이때 선언되는 거 맞나?
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(self.device))       #차원 확인,이때 선언되는 거 맞나?
        torch.nn.init.xavier_uniform_(h_0)
        torch.nn.init.xavier_uniform_(c_0)
        #*Propagate input through LSTM
        x = torch.tensor(x, dtype = torch.float32)
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.num_layers*self.hidden_size)
        out = self.fc1(h_out)
        # out = self.relu(out)
                
        return out

class LSTM3(nn.Module):
    
    '''
    feature 개수 : input_size만큼 반영
    길이 : pred_len만큼 반영
    
    '''
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers, device):
        super(LSTM3, self).__init__()
        self.hist_len = hist_len        #
        self.pred_len = pred_len        #과거 길이, 예측 길이 나눠야됨
        # self.num_features = num_features
        self.num_layers = num_layers
        self.input_size = input_size    
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True, dropout=0.3)
        self.fc1 = nn.Linear(self.hidden_size * self.num_layers, 2*self.hidden_size, bias=True)
        self.fc2 = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.float()
        self.h_0 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)       #차원 확인
        self.c_0 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)
        torch.nn.init.xavier_uniform_(self.h_0)
        torch.nn.init.xavier_uniform_(self.c_0)
    
    def forward(self, x):
        #*Propagate input through LSTM
        x = torch.tensor(x, dtype = torch.float32)
        ula, (h_out, _) = self.lstm(x, (self.h_0, self.c_0))
        h_out = h_out.view(-1, self.num_layers*self.hidden_size)
        out = self.fc1(h_out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # out = self.relu(out)
        
        return out


class DNN1(nn.Module):
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers, device):
        super(DNN1, self).__init__()
        self.hist_len = hist_len        #
        self.pred_len = pred_len
        # self.num_features = num_features
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.fc1 = nn.Linear(self.input_size*self.hist_len, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.float()

    def forward(self, x):
        x = torch.tensor(x, dtype = torch.float32)
        x = x.reshape((-1, self.input_size*self.hist_len))
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # out = self.relu(out)
        
        return out
    

class DNN2(nn.Module):
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers, device):
        super(DNN2, self).__init__()
        self.hist_len = hist_len        #
        self.pred_len = pred_len
        # self.num_features = num_features
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.fc1 = nn.Linear(self.input_size*self.hist_len, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc4 = nn.Linear(self.hidden_size, self.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.float()

    def forward(self, x):
        x = torch.tensor(x, dtype = torch.float32)
        x = x.reshape((-1, self.input_size*self.hist_len))
        out_1 = self.fc1(x)
        out = self.relu(out_1)
        out_2 = self.fc2(out)
        out = out_1 + out_2
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        
        # out = self.relu(out)
        
        return out    


class CNN1(nn.Module):
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers, device):
        super(CNN1, self).__init__()
        # self.conv1 = nn.Conv2d(8, 128, kernel_size=7, bias=True)    #in_channel, out_channel, 
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=24, bias=True)    #in_channel, out_channel, 24시간을 봄
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=8, bias=True)
        self.conv3 = nn.Conv1d(hidden_size, 8, kernel_size=8, bias=True)     
        self.fc1 = nn.Linear(8*131, 128, bias=True)      #!!!! 공식으로 변경하기
        self.fc2 = nn.Linear(128, pred_len, bias=True)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        # # x = x.view(-1, 8, 168)              #!!!!
        x = x.transpose(1,2)
        x = x.type(torch.cuda.FloatTensor).clone().detach().requires_grad_(True)          #! 여기서 메모리 누수 발생하는듯
        # x = torch.tensor(x, dtype = torch.float32)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        # out = out.view(16, -1)              #!!!!
        out = self.relu(self.fc1(out.view(x.shape[0], -1)))
        # out = self.relu(self.fc2(out))
        out = self.fc2(out)
        
        return out


class MultiChannelLSTM(nn.Module):      #* 2 Channel
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers, device):
        super(MultiChannelLSTM, self).__init__()
        self.hist_len = hist_len        #
        self.pred_len = pred_len        #과거 길이, 예측 길이 나눠야됨
        # self.num_features = num_features
        self.num_layers = num_layers
        self.input_size = input_size    
        self.hidden_size = hidden_size
        self.device = device
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True, dropout=0.5)
        self.fc1 = nn.Linear(2*self.hidden_size * self.num_layers, 2*self.hidden_size, bias=True)
        self.fc2 = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, 2*self.hidden_size, bias=True)
        self.out = nn.Linear(2*self.hidden_size, self.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.prelu1 = nn.PReLU()
        # self.prelu2 = nn.PReLU()
        self.dropout = nn.Dropout()
        self.h_1 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)       #Num layers, batch_size, hidden_size
        self.c_1 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)
        self.h_2 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)       #차원 확인
        self.c_2 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)
        torch.nn.init.xavier_uniform_(self.h_1)
        torch.nn.init.xavier_uniform_(self.c_1)
        torch.nn.init.xavier_uniform_(self.h_2)
        torch.nn.init.xavier_uniform_(self.c_2)
        self.float()
        
    def forward(self, x):
        #*Propagate input through LSTM
        x = torch.tensor(x, dtype = torch.float32)
        ula, (h_out1, _) = self.lstm1(x, (self.h_1, self.c_1))
        ula, (h_out2, _) = self.lstm2(x, (self.h_2, self.c_2))
        h_out = torch.cat((h_out1, h_out2), dim=0)                            #hidden concat
        print(h_out)
        h_out = h_out.view(-1, 2*self.num_layers*self.hidden_size)
        print(h_out.shape)
        out_1 = self.fc1(h_out)
        out = self.relu(out_1)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out_3 = self.fc3(out)
        out = out_1 + out_3                 #residual connection
        out = self.out(out)
        
        return out

    
class MultiChannelLSTM_residual_connection(nn.Module):      # 2 Channel #! 미완성
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers, device):
        super(MultiChannelLSTM_residual_connection, self).__init__()
        self.hist_len = hist_len        #
        self.pred_len = pred_len        #과거 길이, 예측 길이 나눠야됨
        # self.num_features = num_features
        self.num_layers = num_layers
        self.input_size = input_size    
        self.hidden_size = hidden_size
        self.device = device
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True, dropout=0.5)
        self.fc1 = nn.Linear(2*self.hidden_size * self.num_layers, 2*self.hidden_size, bias=True, dropout=0.5)
        self.fc2 = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.h_1 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)       #차원 확인
        self.c_1 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)
        self.h_2 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)       #차원 확인
        self.c_2 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)
        torch.nn.init.xavier_uniform_(self.h_1)
        torch.nn.init.xavier_uniform_(self.c_1)
        torch.nn.init.xavier_uniform_(self.h_2)
        torch.nn.init.xavier_uniform_(self.c_2)
        self.float()
        
    def forward(self, x):
        #*Propagate input through LSTM
        x = torch.tensor(x, dtype = torch.float32)
        ula, (h_out1, _) = self.lstm1(x, (self.h_1, self.c_1))
        ula, (h_out2, _) = self.lstm2(x, (self.h_2, self.c_2))
        h_out = torch.cat((h_out1, h_out2), dim=0)                            #hidden concat
        h_out = h_out.view(-1, 2*self.num_layers*self.hidden_size)
        print(h_out.shape)
        out = self.fc1(h_out)
        # out = self.prelu1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.prelu2(out)
        # out = self.relu(out)
        out = self.fc3(out)
        # out = self.relu(out)
        
        return out
    

class MultiChannelLSTM_DifferentTimeScale(nn.Module):       #! 미완성
    
    def __init__(self, hist_len, pred_len, input_size, hidden_size, num_layers, device):
        super(MultiChannelLSTM_DifferentTimeScale, self).__init__()
        self.hist_len = hist_len        #
        self.pred_len = pred_len        #과거 길이, 예측 길이 나눠야됨
        # self.num_features = num_features
        self.num_layers = num_layers
        self.input_size = input_size    
        self.hidden_size = hidden_size
        self.device = device
        self.lstm6 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True, dropout=0.3)
        self.lstm48 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bias=True, dropout=0.3)
        self.fc1 = nn.Linear(2*self.hidden_size * self.num_layers, 2*self.hidden_size, bias=True)
        self.fc2 = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, self.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.h_6 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)       #차원 확인
        self.c_6 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)
        self.h_48 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)       #차원 확인
        self.c_48 = torch.zeros(self.num_layers, 64, self.hidden_size, requires_grad=True).to(self.device)
        torch.nn.init.xavier_uniform_(self.h_6)
        torch.nn.init.xavier_uniform_(self.c_6)
        torch.nn.init.xavier_uniform_(self.h_48)
        torch.nn.init.xavier_uniform_(self.c_48)
        self.float()
        
    def forward(self, x):
        #*Propagate input through LSTM
        x_6 = torch.tensor(x[0], dtype = torch.float32)
        x_48 = torch.tensor(x[1], dtype = torch.float32)
        _, (h_6, _) = self.lstm6(x_6, (self.h_6, self.c_6))
        _, (h_48, _) = self.lstm48(x_48, (self.h_48, self.c_48))
        h_out = torch.cat((h_6, h_48), dim=0)                            #hidden concat
        h_out = h_out.view(-1, 2*self.num_layers*self.hidden_size)
        out = self.fc1(h_out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        # out = self.relu(out)
        
        return out

    
class Attention1(nn.Module):            #! 미완성
    
    def __init__(self):
        super(Attention1, self).__init__()
        
    def forward(self):
        return None


'''
class GRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
'''