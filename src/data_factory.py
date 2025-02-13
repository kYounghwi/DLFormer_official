from src.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS
from torch.utils.data import DataLoader
import pandas as pd
from darts.datasets import ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, WeatherDataset, ExchangeRateDataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'PEMS': Dataset_PEMS,
    'Weather': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'AirQuality': Dataset_Custom,
    'Volatility': Dataset_Custom,
    'SML': Dataset_Custom,
    'PM': Dataset_Custom,
}

# features - Multi-Uni(MS) or Multi-Multi(M) or Uni-Uni(S)
def data_provider(root_path, data, features, batch_size, seq_len, label_len, pred_len, flag):
    Data = data_dict[data]
    
    if data == 'ETTh1':
        df = ETTh1Dataset().load().pd_dataframe()
        df.insert(0, 'date', df.index)
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        target = 'OT'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'ETTh2':
        df = ETTh2Dataset().load().pd_dataframe()
        df.insert(0, 'date', df.index)
        df = df.reset_index(drop=True)
        freq = 'h'
        embed = 'timeF'
        target = 'OT'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'ETTm1':
        df = ETTm1Dataset().load().pd_dataframe()
        df.insert(0, 'date', df.index)
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        target = 'OT'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'ETTm2':
        df = ETTm2Dataset().load().pd_dataframe()
        df.insert(0, 'date', df.index)
        df = df.reset_index(drop=True)
        freq = 'm'
        embed = 'timeF'
        target = 'OT'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'Weather':
        df = WeatherDataset().load().pd_dataframe()  
        df = df.drop(['wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'rain (mm)', 'raining (s)', 'Tpot (K)', 'CO2 (ppm)', 'Tlog (degC)'], axis=1)
        target = 'T (degC)'
        df = df[[col for col in df if col != target] + [target]]
        df.insert(0, 'date', df.index)
        df = df.reset_index(drop=True)
        df = df.dropna()
        freq = 'm'
        embed = 'timeF'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'Exchange':
        df = ExchangeRateDataset().load().pd_dataframe()  
        df.insert(0, 'date', df.index)
        df = df.reset_index(drop=True)
        freq = 'd'
        embed = 'learned'
        target = '7'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'AirQuality':
        df = pd.read_csv(root_path + data + '.csv')
        target = 'NOx(GT)'
        df = df[[col for col in df if col != target] + [target]]
        freq = 'h'
        embed = 'timeF'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'Volatility':
        df = pd.read_csv(root_path + data + '.csv')
        df = df[df['Symbol'] == '.SPX']     # S&P 500
        df = df.drop(['open_time', 'close_time', 'rv5_ss', 'rv10_ss', 'bv_ss', 'rk_twoscale', 'rk_th2',
                      'medrv', 'rsv_ss', 'rv5', 'Symbol'], axis=1)
        target = 'close_price'
        df = df[[col for col in df if col != target] + [target]]
        df.reset_index(drop=True, inplace=True)
        df['date'] = df['date'].str[:-6]
        df['date'] = pd.to_datetime(df['date'])
        freq = 'D'
        embed = 'timeF'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'SML':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.drop(['17:Meteo_Exterior_Sol_Sud', '19:Exterior_Entalpic_1', '20:Exterior_Entalpic_2', '18:Meteo_Exterior_Piranometro', '19:Exterior_Entalpic_1'], axis=1)
        target = '3:Temperature_Comedor_Sensor'
        freq = 'm'
        embed = 'timeF'
        if flag == 'train':
            print(f'Target variable: {target}')
    elif data == 'PM':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df.set_index('date', inplace=True)
        df.insert(0, 'date', df.index)
        target = 'pm2.5'
        freq = 'd'
        embed = 'timeF'
        if flag == 'train':
            print(f'Target variable: {target}')
        
    timeenc = 0 if embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = batch_size  # bsz=1 for evaluation
        freq = freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = batch_size  # bsz for train and valid
        freq = freq

    data_set = Data(
        df=df,
        data_name=data,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq,
    )
    # print(data_set)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)
    
    return data_set, data_loader

def data_select(data, root_path):
    
    if data == 'ETTh1':
        df = ETTh1Dataset().load().pd_dataframe()
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ETTh2':
        df = ETTh2Dataset().load().pd_dataframe()
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ETTm1':
        df = ETTm1Dataset().load().pd_dataframe()
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'ETTm2':
        df = ETTm2Dataset().load().pd_dataframe()
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Weather':
        df = WeatherDataset().load().pd_dataframe()  
        df = df.drop(['wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'rain (mm)', 'raining (s)', 'Tpot (K)', 'CO2 (ppm)', 'Tlog (degC)'], axis=1)
        column_to_move = 'T (degC)'
        df = df[[col for col in df if col != column_to_move] + [column_to_move]]
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Exchange':
        df = ExchangeRateDataset().load().pd_dataframe()  
        freq = 'd'
        embed = 'learned'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'AirQuality':
        df = pd.read_csv(root_path + data + '.csv')
        df.set_index('date', inplace=True)
        target = 'NOx(GT)'
        df = df[[col for col in df if col != target] + [target]]
        freq = 'h'
        embed = 'timeF'
        print(df)
        print(df.columns)
        print(f'freq: {freq} / embed: {embed}')
    elif data == 'Volatility':
        df = pd.read_csv(root_path + data + '.csv')
        df.set_index('date', inplace=True)
        df = df[df['Symbol'] == '.SPX']     # S&P 500
        df = df.drop(['open_time', 'close_time', 'rv5_ss', 'rv10_ss', 'bv_ss', 'rk_twoscale', 'rk_th2',
                      'medrv', 'rsv_ss', 'rv5', 'Symbol'], axis=1)
        column_to_move = 'close_price'
        df = df[[col for col in df if col != column_to_move] + [column_to_move]]
        freq = 'd'
        embed = 'timeF'
        print(df)
        print(df.columns)
    elif data == 'SML':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df = df.drop(['17:Meteo_Exterior_Sol_Sud', '19:Exterior_Entalpic_1', '20:Exterior_Entalpic_2', '18:Meteo_Exterior_Piranometro', '19:Exterior_Entalpic_1'], axis=1)
        df.set_index('date', inplace=True)
        freq = 'm'
        embed = 'timeF'
        print(df)
        print(df.columns)
    elif data == 'PM':
        df = pd.read_csv(root_path + data + '/data' + '.csv')
        df.set_index('date', inplace=True)
        freq = 'd'
        embed = 'timeF'
        print(df)
        print(df.columns)
    
    return df, freq, embed