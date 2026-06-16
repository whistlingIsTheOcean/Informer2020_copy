import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """可选 CNN 前端 — 多层膨胀卷积提取局部时序模式。
    dilations = [1, 2, 4, 8] 覆盖 1+2+4+8=15 步感受野。
    """
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        dilations = [1, 2, 4, 8]
        layers = []
        for d in dilations:
            layers.append(nn.Conv1d(d_model, d_model, kernel_size,
                                    dilation=d, padding=d))
            layers.append(nn.BatchNorm1d(d_model))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):       # [B, L, D]
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class CNNParallel(nn.Module):
    """并行 CNN 分支 — 从原始输入独立提取特征, 与 Attention 编码器输出相加。
    输入 [B, L, enc_in], 输出 [B, L, d_model], 采用残差式合并。
    不使用 BatchNorm (避免破坏时序分布), 用 LayerNorm 替代。
    """
    def __init__(self, enc_in, d_model, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(enc_in, d_model//2, kernel_size=7, padding=3)
        self.ln1 = nn.LayerNorm(d_model//2)
        self.conv2 = nn.Conv1d(d_model//2, d_model, kernel_size=5, padding=2)
        self.ln2 = nn.LayerNorm(d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):       # [B, L, enc_in]
        out = self.conv1(x.transpose(1, 2))            # [B, d/2, L]
        out = self.ln1(out.transpose(1, 2))             # [B, L, d/2]
        out = F.gelu(out)
        out = self.conv2(out.transpose(1, 2))           # [B, d, L]
        out = self.ln2(out.transpose(1, 2))             # [B, L, d]
        out = F.gelu(out)
        out = self.conv3(out.transpose(1, 2))           # [B, d, L]
        out = self.dropout(out.transpose(1, 2))         # [B, L, d]
        return out


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
