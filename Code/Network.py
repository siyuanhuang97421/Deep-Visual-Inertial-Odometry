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

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.num_layers = len(num_channels)
        for i in range(self.num_layers):
            dilation = 2 ** i
            self.conv_layers.append(nn.Conv1d(
                in_channels=self.input_size if i == 0 else num_channels[i-1],
                out_channels=num_channels[i],
                kernel_size=self.kernel_size,
                dilation=dilation
            ))
            self.bn_layers.append(nn.BatchNorm1d(num_channels[i]))
        self.fc1 = nn.Linear(num_channels[-1], 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        for i in range(self.num_layers):
            out = self.conv_layers[i](out)
            out = self.bn_layers[i](out)
            out = nn.functional.relu(out)
            out = nn.functional.dropout(out, p=self.dropout, training=self.training)
        out = out.mean(dim=2)
        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        # split output tensor into translation and quaternion components
        translation = out[:, :3]
        quaternion = out[:, 3:]
        # normalize quaternion components
        quaternion = nn.functional.normalize(quaternion, p=2, dim=1)
        # concatenate translation and quaternion components
        out = torch.cat([translation, quaternion], dim=1)
        return out


# # example input data (batch_size=1, seq_len=100, input_size=6)
# input_data = torch.randn(1, 200, 6)

# # create TCN instance
# tcn = TCN(input_size=6, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2)

# # pass input data to the network
# output = tcn(input_data)

# # print output shape
# print(output.shape)  # should be (batch_size, output_size, n)

    
class DeepVO_nose3(nn.Module):
    def __init__(self):
        super(DeepVO_nose3, self).__init__()

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
        # self.se3_layer = SE3Comp()

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

        # out = self.normalize_quaternion(out)
        pos, quat = out[:, :, :3], out[:, :, 3:]
        quat = nn.functional.normalize(quat, p=2, dim=2)
        out = torch.cat((pos, quat), 2)
        # out = self.se3_layer(out)
        return out
    
    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        return out_conv7
    
    def normalize_quaternion(self, x):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # scale = torch.norm(x[i, j, 3:])
                x[i, j, 3:] = nn.functional.normalize(x[i, j, 3:], p=2, dim=0)


        return x
