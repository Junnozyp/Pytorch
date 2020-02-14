# %%
import os
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.functional as tf
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import init

# To build a simple net with a flatten and a linear layer to classify
# Fashion_Mnist dataset
# please ensure that you have already downloaded the original dataset

def get_fashion_mnist_labels(labels):
    """return true labels according to y_label"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def dataload(path,batch_size,num_workers=None):
    """
    path:upper folder path which contains datasets files
    batch_size:number of one mini_batch
    num_worker: CPU workers when loading image
    return: two iterable object: train_set and test_set
    """
    # build torch.dataset object
    mnist_train = torchvision.datasets.FashionMNIST(root=path, train=True, 
                                            download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=path, train=False, 
                                            download=True, transform=transforms.ToTensor())
    # set num_workeres
    if num_workers is None:
        if sys.platform.startswith('win'):
            num_workers=0
        else:
            num_workers=4
    # build iterable Dataloader object
    train_iter=torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,
                                    num_workers=num_workers)
    test_iter=torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=True,
                                    num_workers=num_workers)
    return train_iter,test_iter,mnist_train,mnist_test

def evaluate_acc(net,data_iter):
        acc_sum,n=0.0,0
        for X,y in data_iter:
            acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
            n+=y.shape[0]
        return acc_sum/n

def train_net(net,train_iter,test_iter,loss,optimizer,lr,epochs,params,ver_gap,verbose=True,test=True):
        for epoch in range(1,epochs+1):
            train_loss_sum,train_acc_sum,n=0.0,0.0,0
            test_acc=0.0
            for X,y in train_iter:
                y_hat=net(X)
                l=loss(y_hat,y).sum()
                
                # clear gradient
                if optimizer is not None:
                    optimizer.zero_grad()
                elif params is not None and params[0].grad is not None:
                    for p in params:
                        # clear grad data of each parameter
                        p.grad.data.zero_()
                
                # bp
                l.backward()
                if optimizer is None:
                    for p in params:
                        p.data-=lr*p.grad/len(y)
                else:
                    optimizer.step()
                
                train_loss_sum+=l.item()
                train_acc_sum+= (y_hat.argmax(dim=1) == y).sum().item()
                n+=y.shape[0]
            if test:
                test_acc=evaluate_acc(net,test_iter)
            if epoch%ver_gap==0 and verbose:
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                    % (epoch, train_loss_sum / n, train_acc_sum / n, test_acc))


#Flatten Layer
class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        return X.view(X.shape[0],-1)


class Linear_Softmax_Net:
    def __init__(self,path,num_workers,num_inputs,num_outputs,learning_rate,batch_size):
        # using sequential to build model easier
        self.net=nn.Sequential(
            OrderedDict([
                ('flatten',FlattenLayer()),
                # insert any layer you need
                ('linear',nn.Linear(num_inputs,num_outputs))
            ])
        )
        # initialize model default parameters
        init.normal_(self.net.linear.weight,mean=0,std=0.01)
        init.constant_(self.net.linear.bias,val=0)
        # load data
        assert os.path.exists(path)
        self.train_iter,self.test_iter,self.mnist_train,_=dataload(path,batch_size,num_workers)
        self.loss=nn.CrossEntropyLoss()
        self.optimizer=torch.optim.SGD(self.net.parameters(),lr=learning_rate,momentum=0.9)
    
    def train(self,epochs,ver_gap,verbose=True,test=True):
        for epoch in range(1,epochs+1):
            train_loss_sum,train_acc_sum,n=0.0,0.0,0
            test_acc=0.0
            for X,y in self.train_iter:
                y_hat=self.net(X)
                l=self.loss(y_hat,y).sum()
                
                # clear gradient
                self.optimizer.zero_grad()
                
                # bp
                l.backward()
                self.optimizer.step()
                
                train_loss_sum+=l.item()
                train_acc_sum+= (y_hat.argmax(dim=1) == y).sum().item()
                n+=y.shape[0]
            if test:
                test_acc=self.evaluate_acc(self.test_iter)
            if epoch%ver_gap==0 and verbose:
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                    % (epoch, train_loss_sum / n, train_acc_sum / n, test_acc))
    
    def evaluate_acc(self,data_iter):
        acc_sum,n=0.0,0
        for X,y in data_iter:
            acc_sum+=(self.net(X).argmax(dim=1)==y).float().sum().item()
            n+=y.shape[0]
        return acc_sum/n

    def save_model(self,model_path):
        assert os.path.exists(model_path)
        torch.save(self.net.state_dict(),model_path)
        print('Saving model successfully')
    
    def load_model_weight(self,model_path):
        assert os.path.exists(model_path)
        self.net.load_state_dict(torch.load(model_path))
        print('Loading model successfully')
    
    # def cross_entropy(self,y_hat,y):
    #     return -torch.log(y_hat.gather(1,y.view(-1,1)))

    def softmax(self,X):
        X_exp=X.exp()
        partition=X_exp.sum(dim=1,keepdim=True)
        return X_exp/partition

    def show_fashion_mnist(self,num):
        # 缺少随机选取的函数
        images,lbls=[],[]
        for img in self.mnist_train:
            if num>0:
                images.append(img[0])
                lbls.append(img[1])
                num-=1
            else: break
        #print(lbls)
        labels=get_fashion_mnist_labels(lbls)
        _,figs=plt.subplots(1,len(images),figsize=(12,12))
        for f,img,lbl in zip(figs,images,labels):
            f.imshow(img.view((28,28)).numpy(),cmap='gray')
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
