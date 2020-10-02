import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ResNet_model import ResNet32_model
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

if __name__ == "__main__":

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

    save_path = "./saved_model/setting_"+str(args.path)+'/epoch_'+str(args.model)+".pth"
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])


    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transforms_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    imgs = ImageFolder('./example', transform=transforms_test)
    print("imgs:", imgs)
    test_loader = DataLoader(imgs, batch_size=1)

    # pp.imshow(inputimg.permute([2, 1, 0]))
    # pp.show()
    print("test_loader:", test_loader)
    print(test_loader.dataset)

    for thisimg, label in test_loader:
        pred = model.forward(thisimg.to(device))
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)
        print("--------------------------------------")
        print("truth:", classes[label])
        print("model prediction:", classes[top_pred])
