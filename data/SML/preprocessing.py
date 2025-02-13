import pandas as pd
import numpy as np
import matplotlib as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# load
data1 = pd.read_csv("data/SML/NEW-DATA-1.T15.txt", sep=' ')
data2 = pd.read_csv("data/SML/NEW-DATA-2.T15.txt", sep=' ')

# 결측값 열 삭제
data1 = data1.drop(columns=['23:Humedad_Exterior_Sensor', '24:Day_Of_Week'])
data2 = data2.drop(columns=['23:Humedad_Exterior_Sensor', '24:Day_Of_Week'])

# date 열 만들기, type 변경
data1['date'] = pd.to_datetime(data1['1:Date'] + ' ' + data1['2:Time'], format='%d/%m/%Y %H:%M')
data1 = data1.drop(columns=['1:Date', '2:Time'])

data2['date'] = pd.to_datetime(data2['1:Date'] + ' ' + data2['2:Time'], format='%d/%m/%Y %H:%M')
data2 = data2.drop(columns=['1:Date', '2:Time'])

# 열 순서 변경 (날짜 앞, target 뒤)
first_column = 'date'
cols = [first_column] + [col for col in data1.columns if col != first_column]
data1 = data1[cols]
last_column = '3:Temperature_Comedor_Sensor'
cols = [col for col in data1.columns if col != last_column] + [last_column]
data1 = data1[cols]

first_column = 'date'
cols = [first_column] + [col for col in data1.columns if col != first_column]
data2 = data2[cols]
last_column = '3:Temperature_Comedor_Sensor'
cols = [col for col in data2.columns if col != last_column] + [last_column]
data2 = data2[cols]

data = pd.concat([data1, data2], axis=0, ignore_index=True)
data.to_csv('data/SML/data.csv', index=False)
