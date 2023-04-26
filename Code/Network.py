import torch
import torch.nn as nn
# from torchsummary import summary
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

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)#maybe remove
        out = self.conv2(out)
        out = self.relu(out)#maybe remove
        out = self.pool(out)
        return out

class PredModel(nn.Module):
    def __init__(self, window_size=200):
        super(PredModel, self).__init__()
        self.conv1 = Conv1DBlock(in_channels=3, out_channels=128, kernel_size=11)
        self.conv2 = Conv1DBlock(in_channels=3, out_channels=128, kernel_size=11)
        self.lstm = nn.LSTM(input_size=256*60, hidden_size=128, num_layers=2,dropout=0.25, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(256, 3)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        # seq_len = x1.size(1)
        # x1 = x1.view(batch_size*seq_len,x1.size(2),x1.size(3))
        # x2 = x2.view(batch_size*seq_len,x2.size(2),x2.size(3))
        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        out = torch.cat((out1, out2), dim=1)
        out = out.view(batch_size,-1)
        out = self.lstm(out)[0]
        y1_pred = self.fc1(out)
        y2_pred = self.fc2(out)
        return y1_pred, y2_pred

# # Instantiate the model
# model = PredModel(window_size=20)
# x1 = torch.rand(1,3,200)
# x2 = torch.rand(1,3,200)
# print(model(x1,x2)[0].shape)

    
class DeepVO(nn.Module):
    def __init__(self):
        super(DeepVO, self).__init__()

        self.batchNorm = True
        self.conv_drop_out = 0.2
        self.conv1   = conv(self.batchNorm,   2,   64, kernel_size=7, stride=2, dropout=self.conv_drop_out)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=self.conv_drop_out)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=self.conv_drop_out)
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=self.conv_drop_out)
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=self.conv_drop_out)
        self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_drop_out)
        self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=self.conv_drop_out)
        self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_drop_out)
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=0.2)
        self.conv7   = conv(self.batchNorm, 1024, 1024, kernel_size=3, stride=2, dropout=0.5)

        self.rnn = nn.LSTM(
            input_size=24576, # ouput dimension of CNN 
            # input_size=98304, # ouput dimension of CNN 
            hidden_size=1000,
            num_layers=2,
            batch_first=True)
        self.rnn.cuda()
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
        out_conv7 = self.conv7(out_conv6)
        return out_conv7