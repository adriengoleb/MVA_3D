#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys

# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
        

        
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),ToTensor()])

def test_transforms():
    return transforms.Compose([ToTensor()])



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}



class MLP(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.MLP1 = nn.Linear(3072, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.MLP2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.MLP3 = nn.Linear(256, classes)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, input):
        hidden = input.reshape(-1, 3072) # equivalent to a flatten

        hidden = self.MLP1(hidden)
        hidden = self.batchnorm1(hidden)
        hidden = F.relu(hidden)

        hidden = self.MLP2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.batchnorm2(hidden)
        hidden = F.relu(hidden)

        hidden = self.MLP3(hidden)
        out = self.activation(hidden)
        return out


class PointNetBasic(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.conv1_1 = nn.Conv1d(3, 64, 1)
        self.batchnorm1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = nn.Conv1d(64, 64, 1)
        self.batchnorm1_2 = nn.BatchNorm1d(64)

        self.conv2_1 = nn.Conv1d(64, 64, 1)
        self.batchnorm2_1 = nn.BatchNorm1d(64)
        self.conv2_2 = nn.Conv1d(64, 128, 1)
        self.batchnorm2_2 = nn.BatchNorm1d(128)
        self.conv2_3 = nn.Conv1d(128, 1024, 1)
        self.batchnorm2_3 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(1024)

        self.MLP3_1 = nn.Linear(1024, 512)
        self.batchnorm3_1 = nn.BatchNorm1d(512)
        self.MLP3_2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm3_2 = nn.BatchNorm1d(256)
        self.MLP3_3 = nn.Linear(256, classes)

        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, input):
        hidden = F.relu(self.batchnorm1_1(self.conv1_1(input)))
        hidden = F.relu(self.batchnorm1_2(self.conv1_2(hidden)))

        hidden = F.relu(self.batchnorm2_1(self.conv2_1(hidden)))
        hidden = F.relu(self.batchnorm2_2(self.conv2_2(hidden)))
        hidden = F.relu(self.batchnorm2_3(self.conv2_3(hidden)))

        hidden = self.maxpool(hidden)
        
        hidden = hidden.reshape(-1, 1024)

        hidden = F.relu(self.batchnorm3_1(self.MLP3_1(hidden)))
        hidden = F.relu(self.batchnorm3_2(self.dropout(self.MLP3_2(hidden))))
        hidden = self.MLP3_3(hidden)
        return self.activation(hidden)
        
        
        
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.conv1_1 = nn.Conv1d(3, 64, 1)
        self.batchnorm1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = nn.Conv1d(64, 128, 1)
        self.batchnorm1_2 = nn.BatchNorm1d(128)
        self.conv1_3 = nn.Conv1d(128, 1024, 1)
        self.batchnorm1_3 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(1024)

        self.MLP2_1 = nn.Linear(1024, 512)
        self.batchnorm2_1 = nn.BatchNorm1d(512)
        self.MLP2_2 = nn.Linear(512, 256)
        self.batchnorm2_2 = nn.BatchNorm1d(256)
        self.MLP2_3 = nn.Linear(256, k*k)

        self.k = k

    def forward(self, input):
        hidden = F.relu(self.batchnorm1_1(self.conv1_1(input)))
        hidden = F.relu(self.batchnorm1_2(self.conv1_2(hidden)))
        hidden = F.relu(self.batchnorm1_3(self.conv1_3(hidden)))

        hidden = self.maxpool(hidden)
        hidden = hidden.reshape(-1, 1024)

        hidden = F.relu(self.batchnorm2_1(self.MLP2_1(hidden)))
        hidden = F.relu(self.batchnorm2_2(self.MLP2_2(hidden)))
        hidden = self.MLP2_3(hidden)

        id3x3 = torch.eye(self.k, requires_grad=True).repeat(hidden.shape[0],1,1)
        if hidden.is_cuda:
            id3x3 = id3x3.cuda()
        # print(np.shape(hidden))
        hidden = hidden.view(-1, self.k, self.k)
        hidden = hidden + id3x3
        hidden = hidden.view(-1, self.k, self.k)

        return hidden


class PointNetFull(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()

        self.tnet = Tnet(k=3)

        self.conv1_1 = nn.Conv1d(3, 64, 1)
        self.batchnorm1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = nn.Conv1d(64, 64, 1)
        self.batchnorm1_2 = nn.BatchNorm1d(64)

        self.conv2_1 = nn.Conv1d(64, 64, 1)
        self.batchnorm2_1 = nn.BatchNorm1d(64)
        self.conv2_2 = nn.Conv1d(64, 128, 1)
        self.batchnorm2_2 = nn.BatchNorm1d(128)
        self.conv2_3 = nn.Conv1d(128, 1024, 1)
        self.batchnorm2_3 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(1024)

        self.MLP3_1 = nn.Linear(1024, 512)
        self.batchnorm3_1 = nn.BatchNorm1d(512)
        self.MLP3_2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm3_2 = nn.BatchNorm1d(256)
        self.MLP3_3 = nn.Linear(256, classes)

        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, input):
        
        tnet_layer = self.tnet(input)

        hidden = torch.matmul(tnet_layer, input)

        hidden = F.relu(self.batchnorm1_1(self.conv1_1(hidden)))
        hidden = F.relu(self.batchnorm1_2(self.conv1_2(hidden)))

        hidden = F.relu(self.batchnorm2_1(self.conv2_1(hidden)))
        hidden = F.relu(self.batchnorm2_2(self.conv2_2(hidden)))
        hidden = F.relu(self.batchnorm2_3(self.conv2_3(hidden)))

        hidden = self.maxpool(hidden)
        
        hidden = hidden.reshape(-1, 1024)

        hidden = F.relu(self.batchnorm3_1(self.MLP3_1(hidden)))
        hidden = F.relu(self.batchnorm3_2(self.dropout(self.MLP3_2(hidden))))
        hidden = self.MLP3_3(hidden)
        return self.activation(hidden), tnet_layer


def basic_loss(outputs, labels):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)



def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    for epoch in range(epochs): 
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.transpose(1,2))
            #outputs, m3x3 = model(inputs.transpose(1,2))
            loss = basic_loss(outputs, labels)
            #loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs = model(inputs.transpose(1,2))
                    #outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))


if __name__ == '__main__':
    
    t0 = time.time()
    
    ROOT_DIR = "../data/ModelNet10_PLY/ModelNet10_PLY" #40
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=default_transforms())
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    model = MLP()
    #model = PointNetBasic()
    #model = PointNetFull()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device);
    
    train(model, device, train_loader, test_loader, epochs = 10)
    
    t1 = time.time()
    print("Total time for training : ", t1-t0)