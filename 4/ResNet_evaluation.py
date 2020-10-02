import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ResNet_model import ResNet32_model
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='4 assignment')
parser.add_argument('--ep', type=int,default=200,help='epoch')
parser.add_argument('--bs', type=int,default=128,help='batchsize')
parser.add_argument('--nl', type=int,default=5,help='num_layers')
parser.add_argument('--lr', type=float,default=0.01,help='learning rate')
parser.add_argument('--ld', type=int,default=0,help='learning rate decay')
parser.add_argument('--mom', type=float,default=0.9,help='momentum')
parser.add_argument('--wd', type=float,default=0,help='weight decay')
parser.add_argument('--da', type=int,default=1,help='data agumentation')
parser.add_argument('--op', type=int,default=1,help='1:SGD, 2:Adam')
parser.add_argument('--gamma', type=float,default=0.1,help='learing rate decay rate')
parser.add_argument('--path', type=int,default=1,help='setting number')
parser.add_argument('--model', type=int,default=1,help='model number')
args = parser.parse_args()

def data_load():
    # CIFAR10 dataset 다운로드
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_data = dsets.CIFAR10(root='./dataset/', train=False, transform=transforms_test, download=True)
    return test_data


def generate_batch(test_data):
    test_batch_loader = DataLoader(test_data, args.bs, shuffle=True)
    return test_batch_loader


def imgshow(image, label, classes):
    print('========================================')
    print(image)
    print('Shape of this image\t:', image.shape)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.title('Label:%s' % classes[label])
    plt.show()
    print('Label of this image:', label, classes[label])


if __name__ == "__main__":
    print('[CIFAR10_evaluation]')
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # GPU 사용시
    if torch.cuda.is_available():
        torch.cuda.device(0)

    # 모델 생성

    model = ResNet32_model()

    if torch.cuda.is_available():
        model = model.to(device)

    model.eval()

    # 데이터 로드
    test_data = data_load()

    # data 개수 확인
    print('The number of test data: ', len(test_data))

    # 배치 생성
    test_batch_loader = generate_batch(test_data)

    # test 시작
    acc_list = []

    # 저장된 state 불러오기
    save_path = "./saved_model/setting_"+str(args.path)+'/epoch_'+str(args.model)+".pth"
    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    correct_cnt = 0
    cnt = 0
    for img, label in test_batch_loader:
        # imgshow(img[1], label[1], classes)
        # quit()
        img = img.to(device)
        label = label.to(device)
        pred = model.forward(img)
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)

        correct_cnt += int(torch.sum(top_pred == label))

    accuracy = correct_cnt / len(test_data) * 100
    print("accuracy of the trained model:%.2f%%" % accuracy)
    acc_list.append(accuracy)


