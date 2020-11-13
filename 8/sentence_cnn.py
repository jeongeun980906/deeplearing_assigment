import numpy as np
import torch
from torch import nn
import argparse
import time

parser = argparse.ArgumentParser(description='8 assignment')
parser.add_argument('--path', type=int,default=1,help='model name')
parser.add_argument('--type', type=int,default=3,help='1: rand 2: static 3: non-static 4: multi-channel')
parser.add_argument('--bs', type=int,default=50,help='mini batch size')
parser.add_argument('--wd', type=float,default=1e-4,help='weight decay')
parser.add_argument('--epoch', type=int,default=100,help='epoch')
parser.add_argument('--data', type=int,default=1,help='1: MR 2:TREC')
parser.add_argument('--lr', type=float,default=0.1,help='learing rate')
parser.add_argument('--act', type=int,default='1',help='activation; 1: relu, 2: elu, 3: leaky_relu')
parser.add_argument('--fs',type=str,default="3,4,5",help='filter size')
parser.add_argument('--fn', type=int,default=100,help='number of filters')
parser.add_argument('--dr', type=float,default=0.5,help='dropout rate')
parser.add_argument('--lc', type=float,default=0.0001,help='l2 constraint')
parser.add_argument('--optimizer', type=int,default=1,help='1 adadelta, 2 adam')
parser.add_argument('--ld', type=int,default=1000,help='learing rate decay for step function')
parser.add_argument('--eds', type=int,default=300,help='embedding dimension size')

args = parser.parse_args()

cpu = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else cpu
filter_list = [int(item)for item in args.fs.split(',')]
if args.data==1:
    data="MR"
elif args.data==2:
    data='TREC'
else:
    print('error')

class ConvFeatures(nn.Module):
    def __init__(self, word_dimension, filter_lengths, filter_counts, dropout_rate):
        super().__init__()
        conv = []
        if args.act==1:
            custom_activation=nn.ReLU()
        elif args.act==2:
            custom_activation=nn.ELU()
        elif args.act==3:
            custom_activation=nn.LeakyReLU()
        
        for size, num in zip(filter_lengths, filter_counts): #filter size 별로 초기화
            conv2d = nn.Conv2d(1, num, (size, word_dimension)) # (input_channel, ouput_channel, height, width)
            nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu') # He initialization
            nn.init.zeros_(conv2d.bias)
            conv.append(nn.Sequential(
                conv2d,
                custom_activation
                #nn.ReLU(inplace=True)
            ))

        self.conv = nn.ModuleList(conv)
        self.filter_sizes = filter_lengths
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embedded_words):
        features = []
        for filter_size, conv in zip(self.filter_sizes, self.conv): #filter size 별로 convolution 수행
            # embedded_words: [batch, sentence length, embedding dimension]
            conv_output = conv(embedded_words)
            conv_output = conv_output.squeeze(-1).max(dim=-1)[0]  # max over-time pooling
            features.append(conv_output)
            del conv_output

        features = torch.cat(features, dim=1) # 각각의 filter에서 나온 feature들을 concatenation
        dropped_features = self.dropout(features)
        return dropped_features


class SentenceCnn(nn.Module):
    def __init__(self, nb_classes, word_embedding_numpy, filter_lengths, filter_counts, dropout_rate):
        super().__init__()

        vocab_size = word_embedding_numpy.shape[0]
        word_dimension = word_embedding_numpy.shape[1]

        # 워드 임베딩 레이어
        self.word_embedding = nn.Embedding(
            vocab_size,
            word_dimension,
            padding_idx=0
        ).to(device)
        # word2vec 활용
        
        if args.type==1: # random
            pass
        
        elif args.type==2: # static
            self.word_embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))
            self.word_embedding.weight.requires_grad = False
        
        elif args.type==3: # non static
            self.word_embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))
        
        elif args.type==4: # multi-channel
            self.word_embedding2 = nn.Embedding(
                vocab_size,
                word_dimension,
                padding_idx=0
                ).to(device)
            self.word_embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))
            self.word_embedding.weight.requires_grad = False
            self.word_embedding2.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))
        
        # 컨볼루션 레이어
        self.features = ConvFeatures(word_dimension, filter_lengths, filter_counts, dropout_rate)

        # 풀리 커텍티드 레이어
        nb_total_filters = sum(filter_counts)
        self.linear = nn.Linear(nb_total_filters, nb_classes).to(device)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, input_x):
        if args.type==4:
            x1 = self.word_embedding(input_x).to(device)
            x1 = x1.unsqueeze(1)
            x1 = self.features(x1)
            x2 = self.word_embedding2(input_x).to(device)
            x2 = x2.unsqueeze(1)
            x2 = self.features(x2)
            x=x1+x2
        else:
            x = self.word_embedding(input_x).to(device)
            x = x.unsqueeze(1)  # 채널 1개 추가
            x = self.features(x)
        
        logits = self.linear(x)
        return logits

class Config():
    def __init__(self):
        self.batch_size = args.bs
        self.lr = args.lr
        self.dropout_rate = args.dr
        self.weight_decay = args.wd
        self.filter_lengths = filter_list
        self.filter_counts = [args.fn]*len(filter_list)
        self.nb_classes = 0
        self.embedding_dim = args.eds
        self.vocab_size = 30000
        self.dev_sample_percentage = 0.1
        self.max_epoch = args.epoch
        self.task = data
        self.mr_train_file_pos = "./data/MR/rt-polarity.pos"
        self.mr_train_file_neg = "./data/MR/rt-polarity.neg"
        self.trec_train_file = "./data/TREC/traindata.txt"
        if args.type==1:
            self.word2vec = None
        else:
            self.word2vec = "./data/GoogleNews-vectors-negative300.bin" # or None

