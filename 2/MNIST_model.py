import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
#python MNIST_train.py  --epoch 30 --Nlayer 3 --layer_size "300,200" --lr 0.001  --wd 0.0 --dropout 0.0 --op Adam --path 1
parser = argparse.ArgumentParser(description='second assignment')
parser.add_argument('--epoch', type=int,default=100,help='number of layers')
parser.add_argument('--Nlayer', type=int,default=3,help='number of layers')
parser.add_argument('--layer_size',type=str,help='layer size')
parser.add_argument('--lr', type=float,default=0.001,help='learning rate')
parser.add_argument('--wi', type=int, default=0,help='weight init')
parser.add_argument('--wd', type=float,default=0,help='weight decay')
parser.add_argument('--dropout', type=float,default=0.0,help='dropout rate')
parser.add_argument('--op', type=str,default='adam',help='Optimizer')
parser.add_argument('--path', type=int,default='1',help='Optimizer')
args = parser.parse_args()
layer_list = [int(item)for item in args.layer_size.split(',')]

class MNIST_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dr=args.dropout
        self.wi=args.wi
        print('weight init',self.wi)
        self.fc1 = nn.Linear(28*28, layer_list[0])
        self.d1=nn.Dropout(self.dr)
        self.fc2 = nn.Linear(layer_list[0], layer_list[1])
        self.d2=nn.Dropout(self.dr)
        if args.Nlayer==3:
            self.fc3 = nn.Linear(layer_list[1], 10)
        elif args.Nlayer==4:
            self.fc3 = nn.Linear(layer_list[1], layer_list[2])
            self.d3=nn.Dropout(self.dr)
            self.fc4 = nn.Linear(layer_list[2], 10)
            if self.wi:
                nn.init.kaiming_normal_(self.fc4.weight.data)
        if self.wi:
            print('1')
            nn.init.kaiming_normal_(self.fc1.weight.data)
            nn.init.kaiming_normal_(self.fc2.weight.data)
            nn.init.kaiming_normal_(self.fc3.weight.data)
    def forward(self, x):
        x = F.relu(self.d1(self.fc1(x)))
        x = F.relu(self.d2(self.fc2(x)))
        if args.Nlayer==3:
            x = F.softmax(self.fc3(x))
        elif args.Nlayer==4:
            x = F.relu(self.d3(self.fc3(x)))
            x = F.softmax(self.fc4(x))
        return x

class Config():
    def __init__(self):
        self.batch_size = 200
        self.lr=args.lr
        self.epoch = args.epoch
        self.weight_decay = args.wd
