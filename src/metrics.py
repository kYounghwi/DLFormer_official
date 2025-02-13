import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from datetime import datetime
from scipy.stats import kendalltau, spearmanr

# metrics
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    pred = pred.squeeze(-1)
    true = true.squeeze(-1)
    
    corrs = []
    for i in range(pred.shape[0]):
        corr = np.corrcoef(pred[i], true[i])[0, 1]
        if not np.isnan(corr).any():  
            corrs.append(corr)
    return np.mean(corrs)

def MAE(pred, true):
    return np.mean(np.abs(true - pred))

def MSE(pred, true):
    return np.mean((true - pred) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true)) * 100

def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true)) * 100

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = np.mean(CORR(torch.tensor(pred), torch.tensor(true)))

    return [mae, mse, rmse, mape, mspe, corr]

# prediction plot
def plot(pred, true, args, now, test=False):

    folder_path = f'results/{args.model_name}/{args.data}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    base_filename = f'{args.model_name}_{args.data}_{args.pred}_{args.seq}_{now.month}{now.day}{now.hour}{now.minute}'
    file_path = os.path.join(os.path.join(folder_path, base_filename), base_filename)

    plt.clf()
    idx = [0, pred.shape[0]*1//4, pred.shape[0]*1//2, pred.shape[0]*3//4]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(pred[idx[0]], label='prediction', color='r')
    axs[0, 0].plot(true[idx[0]], label='actual', color='b')
    axs[0, 0].set_title("0")
    
    axs[0, 1].plot(pred[idx[1]], label='prediction', color='r')
    axs[0, 1].plot(true[idx[1]], label='actual', color='b')
    axs[0, 1].set_title("1/4")
    
    axs[1, 0].plot(pred[idx[2]], label='prediction', color='r')
    axs[1, 0].plot(true[idx[2]], label='actual', color='b')
    axs[1, 0].set_title("1/2")
    
    axs[1, 1].plot(pred[idx[3]], label='prediction', color='r')
    axs[1, 1].plot(true[idx[3]], label='actual', color='b')
    axs[1, 1].set_title("3/4")
    
    plt.tight_layout()
    plt.legend()
    if test:
        plt.savefig(file_path+'_test.png')
    else:
        plt.savefig(file_path+'_val.png')
    

    
    
    
    