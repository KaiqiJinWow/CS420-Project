import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
import argparse
from fer import FER2013
from skimage import io
from skimage.transform import resize
from models import *
import h5py
data = h5py.File('./data/data.h5','r',driver='core')
train_data = data['PrivateTest_pixel']
train_labels = data['PrivateTest_label']
train_data = np.asarray(train_data)
train_data =train_data.reshape((3589,48,48))


parser = argparse.ArgumentParser(description='Attrack Classifier')
parser.add_argument('--model',type=str,default='VGG19',help='CNN architecture')
parser.add_argument('--split', type=str, default='PrivateTest', help='split')
opt = parser.parse_args()
# # Switch to evaluation mode
# net.eval()
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()
path = os.path.join( 'FER2013_' + opt.model)
checkpoint = torch.load(os.path.join(path, opt.split + '_model.t7'),map_location='cpu')

net.load_state_dict(checkpoint['net'])
# net.cuda()
total = train_data.shape[0]
c_pre = 0
c_att = 0
net.eval()
f = open('att.csv', 'a')
f.seek(0)
f.truncate()
for i in range(train_data.shape[0]):
    im_orig = train_data[i] 
    cut_size = 44
    # Remove the mean
    im_orig = im_orig[:, :, np.newaxis]
    img = np.concatenate((im_orig, im_orig, im_orig), axis=2)
    img = Image.fromarray(img)
    # inputs = transform_test(img)
    im = transforms.Compose([
        transforms.CenterCrop(cut_size),
        transforms.ToTensor()
        ]
        )(img)
    #print(im.shape)
    im = im.unsqueeze(0)
    r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    str_label_orig = labels[np.int(label_orig)]
    str_label_pert = labels[np.int(label_pert)]
    print(train_labels[i]) 
    if label_orig == train_labels[i]:
        c_pre = c_pre + 1
    if label_pert == train_labels[i]:
        c_att = c_att + 1
    print("Ground trruth = ", labels[train_labels[i]])
    print("Original label = ", str_label_orig)
    print("Perturbed label = ", str_label_pert)
    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv*torch.ones(A.shape))
        A = torch.min(A, maxv*torch.ones(A.shape))
        return A

    clip = lambda x: clip_tensor(x, 0, 255)
    tf = transforms.Compose([
                            transforms.ToPILImage(),
                            ])
    plt.figure()
    plt.imshow(tf(pert_image.cpu()[0]),cmap = 'gray')
    plt.imshow(pert_image[0])
    image = pert_image.cpu().numpy()
    np.save("att_txt/"+str(i)+".npy",image) 
    plt.title(str_label_orig + " vs " +  str_label_pert)
    plt.savefig("att_image/"+str(i)+".png")
    if (i %100 ==0):
        print(c_pre,c_att)
print(total,c_pre,c_att)
