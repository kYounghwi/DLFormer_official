import torch
import torch.nn as nn
import torch.nn.functional as F

from DLFormer.modules.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from DLFormer.modules.SelfAttention_Family import FullAttention, AttentionLayer, Gated_AttentionLayer
from DLFormer.modules.Embed import DataEmbedding, DataEmbedding_DL

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class Model(nn.Module):
    """
    Distributed Lag Transformer
    """
    def __init__(self, pred_len, seq_len, enc_in, dec_in, c_in, c_out, d_model, d_ff, n_heads, dropout, activation,
                 output_attention, e_layers, d_layers, freq, embed):
        super(Model, self).__init__()
        
        self.pred_len = pred_len
        self.output_attention = output_attention

        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out

        # Embedding
        self.enc_embedding = DataEmbedding_DL(seq_len, c_in, d_model, embed, freq, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=self.output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.dec_embedding = DataEmbedding(self.dec_in, d_model, embed, freq, dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=dropout,
                                      output_attention=self.output_attention),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=self.output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )
            
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        seq_last = x_enc[:, -1:, :].detach()
        # seq_last = torch.mean(x_enc, dim=1).unsqueeze(1).detach()
        x_enc = x_enc - seq_last
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None) 
        # enc_out, [(batch, head, LF, LF) for _ in encoder_layer]
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out, d_attns = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)  
        # dec_out, [(batch, head, Term, LF) for _ in decoder_layer]

        dec_out = dec_out + seq_last[:, :, -1:]

        return dec_out, [attns, d_attns]     
        # dec_out, [[(batch, head, LF, LF) for _ in encoder_layer], [(batch, head, Term, LF) for _ in decoder_layer]]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns  # [B, L, D]
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
# [[LF, LF] for _ in encoder_layer, [Term, LF] for _ in decoder_layer] - average of batch, head
def attn_mean_batchhead(attn): 
    
    for e_layer in range(len(attn[0])):
        attn[0][e_layer] = attn[0][e_layer].mean(dim=(0, 1))
    
    for d_layer in range(len(attn[1])):
        attn[1][d_layer] = attn[1][d_layer].mean(dim=(0, 1))
        
    return attn

# [[LF, LF] for _ in encoder_layer, [Term, LF] for _ in decoder_layer] - average of all sample
def attn_mean_sample(attns):
    
    result = []
    for tensors in zip(*attns):     # each encoder&decoder attention
        averaged_group = []
        for tensor_group in zip(*tensors):  # each layer of encoder&decoder
            stacked = torch.stack(tensor_group, dim=0) 
            averaged = stacked.mean(dim=0)       
            averaged_group.append(averaged.cpu().numpy())
        result.append(averaged_group)
    
    return result

def viz_attn(train_attn, test_attn, args, now, test=None):

    folder_path = f'results/{args.model_name}/{args.data}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    base_filename = f'{args.model_name}_{args.data}_{args.pred}_{args.seq}_{now.month}{now.day}{now.hour}{now.minute}'
    file_path = os.path.join(os.path.join(folder_path, base_filename), base_filename)

    e_layers = args.e_layers
    d_layers = args.d_layers
    if test:
        train = 'Valid'
        test = 'Test'
    else:
        train = 'Train'
        test = 'Valid'

    # Visualization: 2 row(Global, Local), e_layers+d_layers col
    plt.clf()
    fig, axes = plt.subplots(2, len(train_attn[0]) + len(train_attn[1]), figsize=(12, 8))
    fig.suptitle(f"Row: ({train}, {test}) / Col: ({len(train_attn[0])}encoder, {len(train_attn[1])}decoder)", fontsize=14)
    # Train(Val)
    for e_layers in range(len(train_attn[0])):
        im = axes[0, e_layers].imshow(train_attn[0][e_layers], cmap='viridis', aspect='auto')
        axes[0, e_layers].set_title(f"Encoder {e_layers}", fontsize=12)
        axes[0, e_layers].grid(False)
    for d_layers in range(len(train_attn[1])):
        im = axes[0, len(train_attn[0])+d_layers].imshow(train_attn[1][d_layers][-args.pred:], cmap='viridis', aspect='auto')
        axes[0, len(train_attn[0])+d_layers].set_title(f"Decoder {d_layers}", fontsize=14)
        axes[0, len(train_attn[0])+d_layers].grid(False)
    # Val(Test)
    for e_layers in range(len(test_attn[0])):
        im = axes[1, e_layers].imshow(test_attn[0][e_layers], cmap='viridis', aspect='auto')
        axes[1, e_layers].set_title(f"Encoder {e_layers}", fontsize=12)
        axes[1, e_layers].grid(False)
    for d_layers in range(len(train_attn[1])):
        im = axes[1, len(test_attn[0])+d_layers].imshow(test_attn[1][d_layers][-args.pred:], cmap='viridis', aspect='auto')
        axes[1, len(test_attn[0])+d_layers].set_title(f"Decoder {d_layers}", fontsize=12)
        axes[1, len(test_attn[0])+d_layers].grid(False)
        
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.08, label="Value")

    plt.savefig(file_path+f'_{train}{test}_Cross.png')
    
    # preprocessed
    plt.clf()
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))  # 2row 2col
    
    for i in range(2):  
        if i == 0:
            result = np.mean(train_attn[1][0], axis=0)
            result = result.reshape(len(args.cols), args.seq)
        else:
            result = np.mean(test_attn[1][0], axis=0)
            result = result.reshape(len(args.cols), args.seq)
            
        # Temporal importance 
        im = axes[i][0].imshow(result)
        axes[i][0].set_xticks(np.arange(args.seq))
        axes[i][0].set_yticks(np.arange(len(args.cols)))
        axes[i][0].set_xticklabels(["t-" + str(j) for j in np.arange(args.seq - 1, -1, -1)])
        axes[i][0].set_yticklabels(list(args.cols))
        if i == 0:
            axes[i][0].set_title(f"Importance of features and timesteps - {train}", fontsize=16)
        else:
            axes[i][0].set_title(f"Importance of features and timesteps - {test}", fontsize=16)
        cbar = fig.colorbar(im, ax=axes[i][0], orientation='horizontal', fraction=0.02, pad=0.1, label="Importance Scale")
        cbar.ax.tick_params(labelsize=10)
    
        # Variable importance 
        feature_importance = np.mean(result, axis=1)
        axes[i][1].bar(range(len(args.cols)), feature_importance, color="skyblue", edgecolor="black")
        if i == 0:
            axes[i][1].set_title(f"Feature Importance - {train}", fontsize=16)
        else:
            axes[i][1].set_title(f"Feature Importance - {test}", fontsize=16)
        axes[i][1].set_xticks(range(len(args.cols)))
        axes[i][1].set_xticklabels(args.cols)
        axes[i][1].set_ylabel("Importance", fontsize=14)
    
        axes[i][0].grid(False)
        axes[i][1].grid(False)
    
    plt.tight_layout()
    plt.savefig(file_path+f'_{train}{test}_TI_VI.png')
    attn = np.array([train_attn, test_attn], dtype=object)
    np.save(file_path + '_attn.npy', attn)

    

def best(b_model, model, best_metrics, metrics_dict, output_attention, b_train_attn, train_attn, b_test_attn, test_attn):

    best_mae, best_mse, best_rmse, best_mape, best_mspe, best_corr = best_metrics
    mae, mse, rmse, mape, mspe, corr = metrics_dict
    if best_mae > mae: 
        best_mae = mae
        b_model = model
        if output_attention:
            b_train_attn = train_attn
            b_test_attn = test_attn
    if best_mse > mse: best_mse = mse
    if best_rmse > rmse: best_rmse = rmse
    if best_mape > mape: best_mape = mape
    if best_mspe > mspe: best_mspe = mspe
    if best_corr < corr: best_corr = corr
    
    return b_model, [best_mae, best_mse, best_rmse, best_mape, best_mspe, best_corr], b_train_attn, b_test_attn

def save(now, b_model, best_metrics, actual, prediction, args, epoch, b_train_attn, b_test_attn):
    
    metric = np.array(best_metrics)
    results = np.array([metric, actual, prediction, epoch], dtype=object)
    
    folder_path = f'results/{args.model_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    folder_path = f'results/{args.model_name}/{args.data}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    base_filename = f'{args.model_name}_{args.data}_{args.pred}_{args.seq}_{now.month}{now.day}{now.hour}{now.minute}'
    folder_path = os.path.join(folder_path, base_filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_path = os.path.join(folder_path, base_filename)
    np.save(file_path + '.npy', results)
    torch.save(b_model.state_dict(), file_path)    


def log(args, loss, best_metrics, epoch):
    
    if epoch != None:
        print(f'- Data: {args.data} / Horizen: {args.pred} / Lag: {args.seq} / Epoch: {epoch}')
    else:
        print(f'- Data: {args.data} / Horizen: {args.pred} / Lag: {args.seq}')

    if loss != None:
        print(f'Train loss: {loss:.5f}')
        print(f'Val Metric - [MAE: {best_metrics[0]:.3f} / MSE: {best_metrics[1]:.3f} / RMSE: {best_metrics[2]:.3f} / MAPE: {best_metrics[3]:.3f} / MSPE: {best_metrics[4]:.3f} / CORR: {best_metrics[5]:.3f}]')
    else:
        print(f'Test Metric - [MAE: {best_metrics[0]:.3f} / MSE: {best_metrics[1]:.3f} / RMSE: {best_metrics[2]:.3f} / MAPE: {best_metrics[3]:.3f} / MSPE: {best_metrics[4]:.3f} / CORR: {best_metrics[5]:.3f}]')

    











