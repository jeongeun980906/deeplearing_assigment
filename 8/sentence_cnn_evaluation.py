import torch
import smart_open
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sentence_cnn import SentenceCnn, Config
import data_helpers as dh
import argparse

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
parser.add_argument('--ld', type=float,default=0.99,help='learing rate decay for step function')
parser.add_argument('--eds', type=int,default=300,help='embedding dimension size')
args = parser.parse_args()

if __name__ == "__main__":
    print('[CNN for sentence classification evaluation]')
    cfg = Config()

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # GPU 사용시
    if torch.cuda.is_available():
        torch.cuda.device(0)

        # 데이터 로드
    if cfg.task == "MR":
        x_text, test_y = dh.load_mr_data("./data/MR/rt-polarity_test.pos", "./data/MR/rt-polarity_test.neg")
        cfg.nb_classes = max(test_y) + 1
        print("cfg.nb_classes: ", cfg.nb_classes)
    elif cfg.task == "TREC":
        x_text, test_y = dh.load_trec_data("./data/TREC/testdata.txt")
        cfg.nb_classes = max(test_y) + 1
        print("cfg.nb_classes: ", cfg.nb_classes)

    with smart_open.smart_open("./saved_model/vocab"+str(args.path), 'rb') as f:
        word_id_dict = pickle.load(f)
    with smart_open.smart_open("./saved_model/emb"+str(args.path), 'rb') as f:
        initW = pickle.load(f)

    test_x = dh.text_to_indices(x_text, word_id_dict, True)

    # data 개수 확인
    print('The number of test data: ', len(test_x))

    nb_pad = int(max(cfg.filter_lengths) / 2 + 0.5)

    test_x = dh.sequence_to_tensor(test_x, nb_paddings=(nb_pad, nb_pad))
    test_y = torch.tensor(test_y)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # 학습 모델 생성
    model = SentenceCnn(nb_classes=cfg.nb_classes,
                            word_embedding_numpy=initW,
                            filter_lengths=cfg.filter_lengths,
                            filter_counts=cfg.filter_counts,
                            dropout_rate=cfg.dropout_rate).to(device)

    if torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    # test 시작
    acc_list = []

    # 저장된 state 불러오기
    save_path = "./saved_model/setting_"+str(args.path)+"/epoch_"+str(args.epoch)+".pth"
    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    correct_cnt = 0
    cnt = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model.forward(x)
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)
        correct_cnt += int(torch.sum(top_pred == y))

    accuracy = correct_cnt / len(test_x) * 100
    print("accuracy of the trained model:%.2f%%" % accuracy)
    acc_list.append(accuracy)


