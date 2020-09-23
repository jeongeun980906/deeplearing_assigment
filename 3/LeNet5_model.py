import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='third assignment')
parser.add_argument('--lr', type=float,default=0.001,help='learning rate')
parser.add_argument('--ld', type=int,default=0,help='learning rate decay')
parser.add_argument('--wi', type=float, default=0.01,help='weight init 100 for He else float')
parser.add_argument('--wd', type=float,default=0,help='weight decay')
parser.add_argument('--da', type=int,default=0,help='data agumentation')
parser.add_argument('--dr', type=float,default=0.1,help='model number')
parser.add_argument('--path', type=int,default=1,help='setting number')
parser.add_argument('--model', type=int,default=1,help='model number')
args = parser.parse_args()

class LeNet5_model(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################################################################
        #                         TODO : LeNet5 모델 생성                                                             #
        ##############################################################################################################
        # * hint
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.d1    = nn.Dropout(args.dr)
        self.d2    = nn.Dropout(args.dr)
        # max pooling: size: 2, stride 2
        # conv2:  5*5, input_channel: 6, output_channel(# of filters): 16
        # fc1: (16 * 5 * 5, 120)
        # fc2: (120, 84)
        # fc3: (84, 10)
        if args.wi==100:
            nn.init.kaiming_normal_(self.conv1.weight.data)
            nn.init.kaiming_normal_(self.conv2.weight.data)
            nn.init.kaiming_normal_(self.fc1.weight.data)
            nn.init.kaiming_normal_(self.fc2.weight.data)
            nn.init.kaiming_normal_(self.fc3.weight.data)

        elif args.wi<1 and args.wi>0:
            nn.init.constant_(self.conv1.weight.data,args.wi)
            nn.init.constant_(self.conv2.weight.data,args.wi)
            nn.init.constant_(self.fc1.weight.data,args.wi)
            nn.init.constant_(self.fc2.weight.data,args.wi)
            nn.init.constant_(self.fc3.weight.data,args.wi)
        else:
            pass

        # * hint he initialization: stddev = sqrt(2/n), filter에서 n 값은?

    def forward(self, x):
        ##############################################################################################################
        #                         TODO : forward path 수행, 결과를 x에 저장                                            #
        ##############################################################################################################
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.d1(self.fc1(out)))
        out = F.relu(self.d2(self.fc2(out)))
        out = self.fc3(out)
        return out

class Config():
    def __init__(self):
        self.batch_size = 128
        self.lr = args.lr
        self.momentum = 0.9
        self.weight_decay = args.wd
        self.finish_step = 64000
        self.data_augmentation = args.da
