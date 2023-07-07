import os
import sys

from numpy.core.fromnumeric import shape
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from util import config

from datetime import datetime
import numpy as np
import pandas as pd
import arrow
import glob
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.getcwd()




class SolarMetData(data.Dataset):
    
    def __init__(self, flag='Train'):

        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')

        self.fmt = config['dataset']['fmt']
        self.data_start = self._get_time(config['dataset']['data_start'], self.fmt)
        self.data_end = self._get_time(config['dataset']['data_end'], self.fmt)
        self.start_time = self._get_time(config['dataset'][start_time_str], self.fmt)
        self.end_time = self._get_time(config['dataset'][end_time_str], self.fmt)
        
        print(f"data start : {self.data_start}")
        print(f"data end : {self.data_end}")
        print(f"start time : {self.start_time}")
        print(f"end time : {self.end_time}")
        print()
        
        self.solar_list = config['data']['list']
        self.solar_weather_conj = config['data']['solar_weather_conj']
        self.solar_spec = self._load_spec()
        # self.solar_data = self._load_solar()
        # self.solar_data_ratio = self._get_solar_ratio(self.solar_data, self.solar_spec)
        self.weather_var = config['data']['weather_var']
        # self._load_weather()          #create local weather data numpy array
        # self._merge_weather()         #create merged weather data(=total) numpy array
        # self._load_pm()               #create total pm data numpy array
        # self._merge_pm()
        # self._merge_total_data()
        self.total_data = np.load(path + "\\total_solar_weather_pm_2013_2020.npy", allow_pickle=True)
        # self.
        
    def _load_spec(self):
        temp_path = os.getcwd() + '\specification'
        print("Loading Solar Specification Data...\n")
        solar_spec_data = pd.read_csv(temp_path + f"\전국태양광설비정보_v2.csv", encoding='cp949', index_col=0)
        
        return solar_spec_data
    
    
    def _get_time(self, time_yaml, fmt):
        arrow_time = arrow.get(time_yaml, fmt, tzinfo='Asia/Seoul') ###########
        
        return arrow_time
    
    
    def _get_idx(self, t0, t):      #시작점부터 걸린 시간 반환
        # t0 = self.data_start
        return int((t.timestamp() - t0.timestamp()) / (60 * 60))    #Return Hourly Data Index
    
    
    def _get_feature(self):
        weather_var = config['data']['weather_var']
        weather_use = config['experiments']['weather_use']
        weather_idx = [weather_var.index(var) for var in weather_use]
        # self.feature = self.feature[:::::] #slicing!!!!
    
    
    def _load_solar(self):                       # for 비율만 -> 호기는 1개만, 호기 구분은 추후
        # self.solar_list = config['data']['list']      #위에서 정의
        temp_path = os.getcwd() + '\solar_data'
        total_solar = pd.DataFrame(columns=self.solar_list, index=[r for r in arrow.Arrow.range('hour', self.data_start, self.data_end)])  #[col for col in range(self._get_idx(t))])
        
        for col in self.solar_list:
            print(f"Loading solar data of {col}...")
            solar_temp = pd.read_csv(temp_path + f"\\시간대별_태양광_발전량_{col}.csv", encoding='cp949')
            temp_start_time = self._get_time(solar_temp.iloc[0, 2] + " 01", self.fmt)
            temp_end_time = self._get_time(solar_temp.iloc[-1, 2] + " 24", self.fmt)
            temp_start_idx = self._get_idx(self.data_start, temp_start_time)
            temp_end_idx = self._get_idx(self.data_start, temp_end_time)
            # total_solar[temp_start_idx:temp_end_idx+1][col] = np.array(solar_temp.iloc[:,3:]).ravel()     #시간별 데이터 to single column
            total_solar.to_numpy()[temp_start_idx:temp_end_idx+1, self.solar_list.index(col)] = np.array(solar_temp.iloc[:,3:]).ravel() #.astype(np.float32)
            # print(total_solar[temp_start_idx:temp_end_idx+1][col])
        print()
        
        total_solar.to_csv('total_solar_data.csv', encoding='cp949')#, astype=np.float32)

        
        return total_solar
        
    
    def _get_solar_ratio(self, solar_data, solar_spec):
        print('Loading Solar Data Ratio...')
        solar_data_ratio = solar_data
        for col in solar_data_ratio.columns:
            solar_capacity = solar_spec.loc[col, '설비용량']
            temp_solar_data_ratio = solar_data_ratio.to_numpy()[:, solar_data_ratio.columns.tolist().index(col)] / solar_capacity
            solar_data_ratio.to_numpy()[:, self.solar_list.index(col)] = temp_solar_data_ratio
        
        solar_data_ratio.to_csv('total_solar_data_ratio.csv', encoding='cp949')
        
        return solar_data_ratio
        
    
    def _load_weather(self):
        '''
        index 가져와서
        기간에 해당하는 empty array(3차원) 생성하고
        feature 가져오고
        weatherdata 가져오고
        #XXXX pmdata 가져오고 XXXX
        feature에 해당하는 애들만 가져와서
        index 맞게 넣어서(쌓아서?)
        전체 return
        '''
        
        weather_path = os.getcwd() + '\KR_weather\대한민국종관기상관측정보'
        self.weather_station = pd.read_csv(weather_path + '\META_종관기상관측지점정보_중복제거.csv', encoding='cp949')      #나중에 파일 이름 변경
        
        
        
        '''conj 안에서 각 행마다 0열 값 가져오고, 그 행의 1열에 해당하는 이름으로 파일 찾아서, 하나씩 열면서 쌓고, 그것 그대로 반환'''
        total_weather_array = np.empty(shape = (len(self.solar_list), len(self.solar_data_ratio), len(self.weather_var))).astype(object) #nan size 0 위치 / 1 시간 / 2 data
        total_weather_array[:] = np.nan         #data 저장할 size에 해당하는 nan array
               
        for local in self.solar_weather_conj:
            print(f"Loading Weather Data from {local}...")
            weather_loc = local[1]    #config 0, 1, 2 중 1열
            weather_stat_num = self.weather_station['지점'][list(self.weather_station['지점명']).index(f'{weather_loc}')]   #해당하는 위치의 지점 번호
            
            local_weather = pd.DataFrame(columns=self.weather_var, index=[r for r in arrow.Arrow.range('hour', self.data_start, self.data_end)]) #1개 지역에 대해서 저장할 data
            
            '''dataframe으로 만들고, 날짜 일치하는 애들 넣고, 없는 애들 채우고, numpy로 변환해서 return'''
            shift = 0
            year_num = 2013
            for year in glob.glob(os.path.join(weather_path, f"SURFACE_ASOS_{weather_stat_num}*.csv")): 
                print(f"{year}", end=' ')
                yearly_local_weather = pd.DataFrame(columns=self.weather_var, index=[t for t in arrow.Arrow.range('hour', arrow.get('2013-01-01 01').shift(years=shift), arrow.get('2013-12-31 24').shift(years=shift), tz='Asia/Seoul')]) #1개 지역에 대해서 저장할 data
                year_data = pd.read_csv(year, encoding='cp949', index_col=1)                #시간 as index
                print(year_data.shape)
                year_data.index = [arrow.get(t, tzinfo='Asia/Seoul').shift(hours=1) for t in year_data.index]    #solar data와 index match(날씨: 0 측정, solar : 0~1 -> 1)
                
                for time in year_data.index:        #일치하는 시간에 대하여 data 삽입
                    print(f"{local}  :  {time}")
                    if time in yearly_local_weather.index:
                        # yearly_local_weather.to_numpy()[list(yearly_local_weather.index).index(time)] = year_data.loc[time].to_numpy()
                        yearly_local_weather.to_numpy()[self._get_idx(arrow.get('2013-01-01 01', tzinfo='Asia/Seoul').shift(years=shift), time)] = year_data.loc[time].to_numpy()
                        # print(year_data.loc[time].to_numpy())
                        # print(local_weather.to_numpy()[list(local_weather.index).index(time)])
                
                np.save(os.getcwd() + f'\\temp_files\\weather_data_{local[0]}_{year_num}', yearly_local_weather)
                shift += 1
                year_num += 1
                
            print('')


        for local in self.solar_weather_conj:
            solar_loc = local[0]
            for year_num in range(2013, 2021):    
                temp_file = np.load(os.getcwd() + f'\\temp_files\weather_data_{solar_loc}_{year_num}.npy', allow_pickle=True)
                print(temp_file)
                
                if year_num == 2013:
                    local_weather = temp_file
                else:
                    local_weather = np.vstack((local_weather, temp_file))

            np.save(os.getcwd() + f'\\KR_weather\weather_data_{solar_loc}_2013_2020', local_weather)
        
        return 
    
    
    def _merge_weather(self):
        
        local_path = os.getcwd() + '\\KR_weather'
        count = 0
        for local in self.solar_weather_conj:
            print(f"Merging weather file of {local}...")
            # local_year_list = glob.glob(os.path.join(local_path, f"*_{local[0]}_*.npy"))
            temp_file = np.load(local_path + f'\weather_data_{local[0]}_2013_2020.npy', allow_pickle=True)
            temp_file = np.expand_dims(temp_file, axis=0)
            # for year in local_year_list:
            if count == 0:
                total_weather_array = temp_file
            else:
                total_weather_array = np.concatenate((total_weather_array, temp_file), axis=0)
            count += 1
        
        np.save(os.getcwd() + f'\\total_weather_data_2013_2020', total_weather_array)
        
        return 


    def _load_pm(self):
        
        pm_data_path = 'E:\VS_Projects_Graduate\Projects\Fine_Dust_to_Solar_Power\KR_weather\대한민국대기측정최종확정자료\\total_2013_2020'
        pm_list = glob.glob(os.path.join(pm_data_path + '\\*.xlsx'))
        
        pm_total_data = np.array([]).reshape(0,12).astype(object)   #file to save the whole pm data
        
        pm_station_list = dict.fromkeys([pm_station[2][1] for pm_station in config['data']['solar_weather_conj']])
        
        for local_file in pm_list:
            pm_single_data = pd.read_excel(local_file)
        
            for local in pm_station_list:
                print(f"Fetching data of {local} in {local_file[83:]}...")
                local_data = pm_single_data[pm_single_data.iloc[:,2] == local]
                pm_total_data = np.concatenate((pm_total_data, local_data.to_numpy()), axis=0)

        
        pm_total_data = pm_total_data[np.lexsort((pm_total_data[:,4].astype(int), pm_total_data[:,2].astype(int)))]    #sort by station code first, then datetime
        np.save(os.getcwd() + f'\\total_pm_data_2013_2020', pm_total_data)
        
        return
    
    
    def _merge_pm(self):
        pm_data = np.load(os.getcwd() + "\\total_pm_data_2013_2020.npy", allow_pickle=True)
        pm_data = pm_data.reshape(-1, 70128, 12)    #reshape by area followed by year 2013~2020 and features
        pm_station_list = list(dict.fromkeys([i[2][1] for i in config['data']['solar_weather_conj']]))
        pm_station_list.sort()      #to find with index in pm_data
        
        total_local_pm_data = np.array([]).reshape(0,70128,12).astype(object)
        for local in self.solar_weather_conj:
            print(f"Merging pm data for each {local}...")
            pm_code = local[2][1]
            # pm_station_list.index(pm_code)            
            total_local_pm_data = np.concatenate((total_local_pm_data, np.expand_dims(pm_data[pm_station_list.index(pm_code)], axis=0)), axis=0)
    
        np.save(os.getcwd() + f'\\total_local_pm_data_2013_2020', total_local_pm_data)
        
    
    def _merge_total_data(self):
        
        total_pm_data = np.load(os.getcwd() + "\\total_local_pm_data_2013_2020.npy", allow_pickle=True)
        total_weather_data = np.load(os.getcwd() + "\\total_weather_data_2013_2020.npy", allow_pickle=True)
        total_solar_ratio = pd.read_csv(os.getcwd() + '\\total_solar_data_ratio.csv', encoding='cp949', index_col=0)
        total_solar_ratio = np.expand_dims(total_solar_ratio[:70128].T.to_numpy(), axis=2)
        total_data = np.concatenate((total_solar_ratio, total_weather_data, total_pm_data), axis=2)
        
        np.save(os.getcwd() + f'\\total_solar_weather_pm_2013_2020', total_data)
        
        return
    
    
    
'''
fmt = "YYYY-MM-DD-HH-mm"
start_time = arrow.get("2015-01-01")#, fmt)
end_time = arrow.get("2016-01-01")
# end_time = [[2016, 12, 31], GMT]
# data_start = [[2015, 1, 1, 0, 0], GMT]
# data_end = [[2018, 12, 31, 21, 0], GMT]
(end_time.timestamp() - start_time.timestamp()) / (60 * 60)
'''

data = SolarMetData()

# data.solar_data.to_csv('total_solar_data.csv', encoding='cp949')#, astype=np.float32)
# data.solar_data_ratio.to_csv('total_solar_data_ratio.csv', encoding='cp949')
# np.save(os.getcwd() + '\\weather_data', data.weather_data)

# print(data.weather_data.shape)
# print(data.solar.dropna())


'''
raw = {'hpt_res': train_metadata['hpt_res'],
        'features' : train_metadata['features'],
        'best_model' : train_metadata['best_model'],
        'search_method' : train_metadata['search_method'],
        'error_method' : train_metadata['error_method']}
data = data.append(raw, ignore_index = True)
'''  