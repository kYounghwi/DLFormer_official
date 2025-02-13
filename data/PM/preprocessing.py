import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# load
df = pd.read_csv("data/PM/PRSA_data_2010.1.1-2014.12.31.csv")
df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df = df.drop(columns=['No', 'year', 'month', 'day', 'hour', 'cbwd'])
df = df[['date'] + [col for col in df.columns if col != 'date']]
df = df.dropna()

column_to_move = 'pm2.5'
df = df[[col for col in df.columns if col != column_to_move] + [column_to_move]]
ratio = int(df.shape[0]*0.8)
df = df.iloc[:ratio]

for i in range(1, df.shape[1]):
    plt.figure(figsize=(20, 12))
    plt.plot(df.iloc[:, 0], df.iloc[:, i].values)
    plt.title(f'{df.columns[i]}')
    plt.show()
    
print(df)

df.to_csv('data/PM/data.csv', index=False)


