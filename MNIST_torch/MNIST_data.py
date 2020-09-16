import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from MNIST_torch.MNIST_model import *

def data_load():

    # MNIST dataset 다운로드
    train_data = dsets.MNIST(root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
    val_data = dsets.MNIST(root="./dataset/", train=False, transform=transforms.ToTensor(), download=True)

    return train_data, val_data


def imgshow(image, label):
    print('========================================')
    print("The 1st image:")
    print(image)
    print('Shape of this image\t:', image.shape)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title('Label:%d' % label)
    plt.show()
    print('Label of this image:', label)

if __name__ == '__main__':
    # configuration
    cfg = Config()

    # 데이터 로드
    # MNIST datset: 28 * 28 사이즈의 이미지들을 가진 dataset
    train_data, val_data = data_load()

    # data 개수 확인
    print('The number of training data: ', len(train_data))
    print('The number of validation data: ', len(val_data))

    # shape 및 실제 데이터 확인
    image, label = train_data[0]
    imgshow(image, label)
