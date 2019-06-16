'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--model1', type=str, default='Resnet18', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--test', action = 'store_true', help ='test dataset')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 250

dataset = 'FER2013'

path = os.path.join("models",dataset + '_' + opt.model)
path1 = os.path.join("models",dataset + '_' + opt.model1)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()

if opt.model1 == 'VGG19':
    net1 = VGG('VGG19')
elif opt.model1  == 'Resnet18':
    net1 = ResNet18()


if opt.test:
    checkpoint = torch.load(os.path.join(path, 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    checkpoint = torch.load(os.path.join(path1, 'PrivateTest_model.t7'))
    net1.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']


if use_cuda:
    net.cuda()
    net1.cuda()

def Test_survey(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    net1.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    TP,TN,FN,FP = [0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        #print(inputs.shape)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)*torch.Tensor([0.487625662,0.479166667,0.498931624,0.497438951,0.508753814,0.500770712,0.490041861]).cuda()  # avg over crops
        outputs1 = net1(inputs)
        outputs_avg1 = outputs1.view(bs, ncrops, -1).mean(1)*torch.Tensor([0.512374338,0.520833333,0.501068376,0.502561049,0.491246186,0.499229288,0.509958139]).cuda()
        outputs_avg = torch.add(outputs_avg,outputs_avg1)
        #print(outputs_avg)
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.data[0]
        _, predicted = torch.max(outputs_avg.data, 1)
        #print(predicted,targets)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        for i in range(targets.size(0)):
            for j in range(7):
                TP[j] += ((predicted[i] == j) & (targets.data[i] == j)).cpu().sum()
                TN[j] += ((predicted[i] != j) & (targets.data[i] != j)).cpu().sum()
                FN[j] += ((predicted[i] != j) & (targets.data[i] == j)).cpu().sum()
                FP[j] += ((predicted[i] == j) & (targets.data[i] != j)).cpu().sum()
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total
    print(TP,TN,FN,FP)
    for i in range(7):
        print(TP[i].item()/(TP[i].item()+FP[i].item()))
        print(TP[i].item()/(TP[i].item()+FN[i].item()))
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
            'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

if not(opt.test):
    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        PublicTest(epoch)
        PrivateTest(epoch)

    print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
    print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
    print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
    print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)

if opt.test:
    Test_survey(1)

