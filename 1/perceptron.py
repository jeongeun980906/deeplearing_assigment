import argparse
import torch
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description='first assignment')
parser.add_argument('input', nargs='+', type=float,help='3 inputs of perceptron')
args = parser.parse_args()

x_data=args.input
#x_data = np.array([[0.1, 2.3, 1.5]], dtype=np.float32)
x_data = torch.FloatTensor(x_data).unsqueeze(0)

#print(x_data)
linear = nn.Linear(3, 1)

sign_hypothesis = torch.sign(linear(x_data))
sigmoid_hypothesis = torch.sigmoid(linear(x_data))
relu_hypothesis = torch.relu(linear(x_data))

print('step:{}, sigmoid:{}, ReLU:{}'.format(sign_hypothesis.item(),\
            sigmoid_hypothesis.item(),relu_hypothesis.item()))