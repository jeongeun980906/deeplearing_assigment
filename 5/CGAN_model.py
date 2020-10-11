import os
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
##############################################################################################################
#                    TODO : X1 ~ X7에 올바른 숫자 또는 변수를 채워넣어 CGAN 코드를 완성할 것                 #
##############################################################################################################


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes) # 각각의 label을 표현하는 features를 위한 weight matrix
        
        self.fc = nn.Sequential(
            nn.Linear(100+10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 *7 *7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )
        # self.model = nn.Sequential(
        #     nn.Linear(100 + 1, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, int(np.prod(self.img_shape))),
        #     nn.Tanh()
        # )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        # gen_input = torch.cat((noise,labels), -1)
        # img = self.model(gen_input)
        # img = img.view(img.size(0), *self.img_shape) # generation한 image를 reshape
        labels=self.label_emb(labels).squeeze(1)
        img = torch.cat([noise, labels], 1)
        img=self.fc(img)
        img=img.view(-1,128,7,7)
        img=self.deconv(img)
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, 28*28) # 각각의 label을 표현하는 features를 위한 weight matrix
        #self.label_embedding = nn.Embedding(n_classes, n_classes)
        # self.model = nn.Sequential(
        #     nn.Linear(28*28 + 10, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid(),
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7* 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
#        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels.squeeze(-1))), -1)
#        validity = self.model(d_in)
        labels=self.label_embedding(labels)
        labels=labels.view(-1,1,28,28)
        validity = torch.cat([img, labels], 1)
        validity=self.conv(validity)
        validity=validity.view(-1,128*7*7)
        validity=self.fc(validity)
        return validity

class Config():
    def __init__(self):
        self.batch_size = 128
        self.lr = 0.0005
        self.num_epochs = 50
        self.latent_dim = 100 # noise vector size
        self.n_classes = 10
        self.img_size = 28
        self.channels = 1