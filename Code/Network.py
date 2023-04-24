import torch
import torch.nn as nn
import numpy as np
from se3_layer import *

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )

class DeepIO(nn.Module):
    def __init__(self):
        super(DeepIO, self).__init__()
        self.rnnIMU = nn.LSTM(
            input_size=6, 
            hidden_size=1000,
            num_layers=2,
            batch_first=True)
        self.rnnIMU.cuda()
        self.rnn_drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=1000, out_features=7)
        self.se3_layer = SE3Comp()


    def forward(self, imu):
        out, (imu_n, imu_c) = self.rnnIMU(imu)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        out = self.se3_layer(out)
        # out = imu_out[:, -1, :]
        #print('imu_out', imu_out.shape)
        # imu_out = imu_out.unsqueeze(1)

        return out
    
class DeepVO(nn.Module):
    def __init__(self):
        super(DeepVO, self).__init__()

        self.batchNorm = True
        self.conv_drop_out = 0.2
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=self.conv_drop_out)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=self.conv_drop_out)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=self.conv_drop_out)
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=self.conv_drop_out)
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=self.conv_drop_out)
        self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_drop_out)
        self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=self.conv_drop_out)
        self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_drop_out)
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=0.5)

        self.rnn = nn.LSTM(
            input_size=6, 
            hidden_size=1000,
            num_layers=2,
            batch_first=True)
        self.rnnIMU.cuda()
        self.rnn_drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=1000, out_features=7)
        self.se3_layer = SE3Comp()

        # weights initialization

    def forward(self, x):
        # x: (batch, seq_len, channel, width, height)
        # stack_image
        x = torch.cat(( x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)

        # RNN
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        return out
    
    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6