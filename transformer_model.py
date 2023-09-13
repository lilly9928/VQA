import os
import time
import re
import math
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
import torch.nn
import torchvision.models as models
import datetime
import torchvision
import math
from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange,Reduce
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from torchsummary import summary



class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x

class ImgEncoder(nn.Module):
    def __init__(self, in_channel:int =3, patch_size: int = 16, emb_size:int = 768, img_size:int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channel,emb_size,kernel_size=patch_size,stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x


class PositionalEncoding(nn.Module):
    def __init__(self,dim_model,dropput_p,max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropput_p)
        pos_encoding = torch.zeros(max_len,dim_model)
        position_list = torch.arange(0,max_len,dtype=torch.float).view(-1,1)
        division_term = torch.exp(torch.arange(0,dim_model,2).float()*(-math.log(10000.0)/dim_model))
        pos_encoding[:,0::2] = torch.sin(position_list * division_term)
        pos_encoding[:,1::2] = torch.cos(position_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0,1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self,token_embedding:torch.tensor)->torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0),:])

class QstEncoder(nn.Module):

    def __init__(self,num_tokens,dim_model,num_head,num_encoder_layers,dropout_p):
        super().__init__() #왜해주는지 알아보기
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropput_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
    def forward(self,qst):
        x = self.embedding(qst)*math.sqrt(self.dim_model)
        x = self.positional_encoder(x)
        x = x.permute(1,0,2)

        return x


class VQAModel(nn.Module):

    def __init__(self,in_channel:int =3, patch_size: int = 16, emb_size:int = 768, img_size:int = 224,
                 num_tokens:int =4,dim_model:int =8,num_heads:int =2,num_encoder_layers:int =3,dropout_p=0.1):
        super(VQAModel,self).__init__()
        self.img_encoder = ImgEncoder(in_channel,patch_size,emb_size,img_size)
        self.qst_encoder = QstEncoder(num_tokens,dim_model,num_heads,num_encoder_layers,dropout_p)
    def forward(self,img,qst):
        img_feature = self.img_encoder(img)
        qst_feature = self.qst_encoder(qst)

        return img_feature,qst_feature



if __name__ == '__main__':

    input_dir = 'D:/data/vqa/coco/simple_vqa'
    log_dir = './logs'
    model_dir='./models'
    max_qst_length = 30
    max_cap_length=50
    max_num_ans =10
    embed_size=64
    word_embed_size=300
    num_layers=2
    hidden_size=32
    learning_rate = 0.001
    step_size = 10
    gamma = 0.1
    num_epochs=30
    batch_size = 16
    num_workers = 4
    save_step=1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQAModel().to(device)
    summary(model, [(3, 224, 224),(16,30)], device="cuda")
