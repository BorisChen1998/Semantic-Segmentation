from  cityscape  import DatasetTrain ,DatasetVal, DatasetTest
import argparse
import torch
from torch.utils.data import  DataLoader
from pathlib import Path
import yaml
from train import Trainer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from model.BLNet import Net
from config import Config
from loss import CrossEntropyLoss2d, LovaszSoftmax
import numpy as np

weight = torch.ones(20)
'''
weight[0] = 2.3653597831726	
weight[1] = 4.4237880706787	
weight[2] = 2.9691488742828	
weight[3] = 5.3442072868347	
weight[4] = 5.2983593940735	
weight[5] = 5.2275490760803	
weight[6] = 5.4394111633301	
weight[7] = 5.3659925460815	
weight[8] = 3.4170460700989	
weight[9] = 5.2414722442627	
weight[10] = 4.7376127243042	
weight[11] = 5.2286224365234	
weight[12] = 5.455126285553	
weight[13] = 4.3019247055054	
weight[14] = 5.4264230728149	
weight[15] = 5.4331531524658	
weight[16] = 5.433765411377	
weight[17] = 5.4631009101868	
weight[18] = 5.3947434425354
'''
weight[0] = 2.8149201869965	
weight[1] = 6.9850029945374	
weight[2] = 3.7890393733978	
weight[3] = 9.9428062438965	
weight[4] = 9.7702074050903	
weight[5] = 9.5110931396484	
weight[6] = 10.311357498169	
weight[7] = 10.026463508606	
weight[8] = 4.6323022842407	
weight[9] = 9.5608062744141	
weight[10] = 7.8698215484619	
weight[11] = 9.5168733596802	
weight[12] = 10.373730659485	
weight[13] = 6.6616044044495	
weight[14] = 10.260489463806	
weight[15] = 10.287888526917	
weight[16] = 10.289801597595	
weight[17] = 10.405355453491	
weight[18] = 10.138095855713

weight[19] = 0

# python3 setup.py install --plugins --cuda-dir=/usr/local/cuda/ --torch-dir=/home/chenxiaoshuang/anaconda3/envs/pt/lib/python3.7/site-packages/torch/ --trt-lib-dir=/home/chenxiaoshuang/anaconda3/envs/pt/lib/python3.7/site-packages/tensorrt/ --trt-inc-dir=/home/chenxiaoshuang/TensorRT-7.0.0.11/lib
# ldd -r /home/chenxiaoshuang/anaconda3/envs/pt/lib/python3.7/site-packages/torch2trt/libtorch2trt.so

if __name__=='__main__':

    cfg=Config()
    #create dataset
    train_dataset = DatasetTrain(cityscapes_data_path="/home/chenxiaoshuang/cityscapes",
                                cityscapes_meta_path="/home/chenxiaoshuang/cityscapes/gtFine")
    val_dataset = DatasetVal(cityscapes_data_path="/home/chenxiaoshuang/cityscapes",
                             cityscapes_meta_path="/home/chenxiaoshuang/cityscapes/gtFine")
    test_dataset = DatasetTest(cityscapes_data_path="/home/chenxiaoshuang/cityscapes",
                             cityscapes_meta_path="/home/chenxiaoshuang/cityscapes/gtFine")      
    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=8, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset,
                                         batch_size=8, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset,
                                         batch_size=8, shuffle=False, num_workers=8)

    net = Net(num_classes=20)
    '''
    net_lite = Net_lite(num_classes=20)
    net = net.cuda()
    net_lite = net_lite.cuda()
    checkpoint = torch.load('./log/dfanet20200207T1900/model_dfanet_0000.pt')
    net_lite.load_state_dict(checkpoint['model_state_dict'])
    net.encoder = net_lite.encoder
    torch.save({
                'epoch': 0,
                'model_state_dict': net.state_dict(),
                #'optimizer_state_dict': self.optim.state_dict(),
                #'lr_scheduler': self.scheduler.state_dict(),
                #'loss': loss
            }, "./log/dfanet20200211T1319/model_dfanet_0000.pt")
    '''
    if torch.cuda.is_available():
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)
    #criterion = LovaszSoftmax(weight=weight)
    
    optimizer = optim.Adam(net.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    lambda1 = lambda epoch : (1 - epoch/300) ** 0.9

    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)
    
    trainer = Trainer('training', optimizer, exp_lr_scheduler, net, cfg, './log')
    trainer.load_weights(trainer.find_last())
    trainer.train(train_loader, val_loader, criterion, 300)
    #trainer.evaluate(val_loader)
    #trainer.fps(val_loader)
    #trainer.test(test_loader)

    
    print('Finished Training')
