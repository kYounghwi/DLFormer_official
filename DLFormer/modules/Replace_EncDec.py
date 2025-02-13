import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer_FFN(nn.Module):
    def __init__(self, DL_size, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_FFN, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Replace FFN
        self.conv3 = nn.Conv1d(in_channels=DL_size, out_channels=DL_size*2, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=DL_size*2, out_channels=DL_size, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        # DL FFN
        x_ = self.dropout(self.activation(self.conv3(x)))
        x_ = self.dropout(self.conv4(x_))
        x_ = self.norm1(x_ + x)
        
        # Embedding FFN
        y = self.dropout(self.activation(self.conv1(x_.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(y + x_)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D] iT [B N E]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)

        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        return x

class DecoderLayer_FFN(nn.Module):
    def __init__(self, pred_len, DL_size, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer_FFN, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Self FFN
        self.conv3 = nn.Conv1d(in_channels=pred_len, out_channels=pred_len*2, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=pred_len*2, out_channels=pred_len, kernel_size=1)
        # Cross FFN
        self.conv5 = nn.Conv1d(in_channels=DL_size, out_channels=DL_size*2, kernel_size=1)
        self.conv6 = nn.Conv1d(in_channels=DL_size*2, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        
        x_ = self.dropout(self.activation(self.conv3(x)))
        x_ = self.dropout(self.conv4(x_))
        x_ = self.norm1(x_ + x)
        
        scores = torch.einsum("bpe,bde->bpd", x_, cross)
        
        s = self.dropout(self.activation(self.conv5(scores.transpose(-1, 1))))
        s = self.dropout(self.conv6(s)).transpose(-1, 1)
        y = s = self.norm2(s + x_)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(s + y)
    
class DecoderLayer_Self_FFN(nn.Module):
    def __init__(self, pred_len, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer_Self_FFN, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Self FFN
        self.conv3 = nn.Conv1d(in_channels=pred_len, out_channels=pred_len*2, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=pred_len*2, out_channels=pred_len, kernel_size=1)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        
        x_ = self.dropout(self.activation(self.conv3(x)))
        x_ = self.dropout(self.conv4(x_))
        x_ = self.norm1(x_ + x)

        inp, attns = self.cross_attention(x_, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)
        # inp, [(batch, head, Term, LF) for _ in decoder_layer]
        x_ = x_ + self.dropout(inp)
        x_ = self.norm2(x_)
        
        y = self.dropout(self.activation(self.conv1(x_.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(y + x_)

class DecoderLayer_Cross_FFN(nn.Module):
    def __init__(self, self_attention, DL_size, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer_Cross_FFN, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Cross FFN
        self.conv5 = nn.Conv1d(in_channels=DL_size, out_channels=DL_size*2, kernel_size=1)
        self.conv6 = nn.Conv1d(in_channels=DL_size*2, out_channels=d_model, kernel_size=1)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        scores = torch.einsum("bpe,bde->bpd", x, cross)
        
        s = self.dropout(self.activation(self.conv5(scores.transpose(-1, 1))))
        s = self.dropout(self.conv6(s)).transpose(-1, 1)
        y = s = self.norm2(s + x)
        
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(y + s)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        attns = []
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
