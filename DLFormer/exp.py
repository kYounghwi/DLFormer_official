# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:33:37 2025

@author: Younghwi Kim
"""
import torch
import DLFormer.modules.DLFormer as DLFormer
import src.metrics as metrics

import warnings
warnings.filterwarnings("ignore")

def build_model(args):
    
    model = DLFormer.Model(pred_len=args.pred, seq_len=args.seq, enc_in=args.e_layers, dec_in=args.d_layers, c_in=args.n_features, c_out=1,
                           d_model=args.d_model, d_ff=args.d_ff, n_heads=args.heads, dropout=args.dropout, 
                           activation=args.activation, output_attention=args.output_attention,
                           e_layers=args.e_layers, d_layers=args.d_layers, freq=args.freq, embed=args.embed).float().to(args.device)
    return model

def train(args, model, criterion, optimizer, train_loader, val_loader, now):
    
    device = args.device
    best_metrics = [10**9, 10**9, 10**9, 10**9, 10**9, -1*10**9]
    b_model, b_train_attn, b_test_attn = None, None, None
    
    for epoch in range(args.num_epochs):
        model.train()
        train_attn = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()
            batch_y = batch_y[:, :, -1:].float().to(device)
            dec_inp = torch.zeros([batch_y.size(0), args.pred, batch_y.shape[-1]]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
            batch_x, batch_x_mark, batch_y_mark = batch_x.float().to(device), batch_x_mark.float().to(device), batch_y_mark.float().to(device)
            
            if args.output_attention:
                output, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  
                train_attn.append(DLFormer.attn_mean_batchhead(attn))
            else:
                output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            loss = criterion(output, batch_y[:, -args.pred:])
            loss.backward()
            optimizer.step()

        if args.output_attention:
            train_attn = DLFormer.attn_mean_sample(train_attn)
        
        model.eval()
        actual, prediction, test_attn = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_y = batch_y[:, :, -1:].float().to(device)
                dec_inp = torch.zeros([batch_y.size(0), args.pred, batch_y.shape[-1]]).float().to(device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
                batch_x, batch_x_mark, batch_y_mark = batch_x.float().to(device), batch_x_mark.float().to(device), batch_y_mark.float().to(device)
                
                if args.output_attention:
                    output, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  
                    test_attn.append(DLFormer.attn_mean_batchhead(attn))
                else:
                    output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                actual.append(batch_y[:, -args.pred:])
                prediction.append(output)
            
            if args.output_attention:
                test_attn = DLFormer.attn_mean_sample(test_attn)
            
        actual = torch.cat(actual, dim=0).cpu().numpy()
        prediction = torch.cat(prediction, dim=0).cpu().numpy()
        
        metrics_dict = metrics.metric(prediction, actual)
        b_model, best_metrics, b_train_attn, b_test_attn = DLFormer.best(b_model, model, 
                                                                           best_metrics, metrics_dict, 
                                                                           args.output_attention, 
                                                                           b_train_attn, train_attn, 
                                                                           b_test_attn, test_attn)
        DLFormer.save(now, b_model, best_metrics, actual, prediction, args, epoch, b_train_attn, b_test_attn)
        
        if epoch%args.log_interval == 0:
            DLFormer.log(args, loss, best_metrics, epoch)
        if epoch%args.plot_interval == 0:   
            metrics.plot(prediction, actual, args, now)
            if args.output_attention:
                DLFormer.viz_attn(train_attn, test_attn, args, now)
    
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    
def test(args, criterion, optimizer, test_loader, now):

    device = args.device
    model_path = f'{args.model_name}_{args.data}_{args.pred}_{args.seq}_{now.month}{now.day}{now.hour}{now.minute}'
    model = DLFormer.Model(pred_len=args.pred, seq_len=args.seq, enc_in=args.e_layers, dec_in=args.d_layers, c_in=args.n_features, c_out=1,
                           d_model=args.d_model, d_ff=args.d_ff, n_heads=args.heads, dropout=args.dropout, 
                           activation=args.activation, output_attention=args.output_attention,
                           e_layers=args.e_layers, d_layers=args.d_layers, freq=args.freq, embed=args.embed).float().to(device)
    model.load_state_dict(torch.load(f'results/{args.model_name}/{args.data}/{model_path}/{model_path}'))

    model.eval()
    
    val_attn = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_y = batch_y[:, :, -1:].float().to(device)
            dec_inp = torch.zeros([batch_y.size(0), args.pred, batch_y.shape[-1]]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
            batch_x, batch_x_mark, batch_y_mark = batch_x.float().to(device), batch_x_mark.float().to(device), batch_y_mark.float().to(device)
            
            if args.output_attention:
                output, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  
                val_attn.append(DLFormer.attn_mean_batchhead(attn))
            else:
                output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
        if args.output_attention:
            val_attn = DLFormer.attn_mean_sample(val_attn)
    
    actual, prediction, test_attn = [], [], []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_y = batch_y[:, :, -1:].float().to(device)
            dec_inp = torch.zeros([batch_y.size(0), args.pred, batch_y.shape[-1]]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
            batch_x, batch_x_mark, batch_y_mark = batch_x.float().to(device), batch_x_mark.float().to(device), batch_y_mark.float().to(device)
            
            if args.output_attention:
                output, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  
                test_attn.append(DLFormer.attn_mean_batchhead(attn))
            else:
                output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
            actual.append(batch_y[:, -args.pred:])
            prediction.append(output)
        
        if args.output_attention:
            test_attn = DLFormer.attn_mean_sample(test_attn)
        
    actual = torch.cat(actual, dim=0).cpu().numpy()
    prediction = torch.cat(prediction, dim=0).cpu().numpy()
    
    metrics_dict = metrics.metric(prediction, actual)
    metrics.plot(prediction, actual, args, now, True)
    if args.output_attention:
        DLFormer.viz_attn(val_attn, test_attn, args, now, test=True)
    DLFormer.log(args, None, metrics_dict, None)
    
    torch.cuda.empty_cache()
    import gc
    gc.collect()