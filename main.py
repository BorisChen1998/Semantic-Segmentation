from cityscape import DatasetTrain ,DatasetVal, DatasetTest
from camvid import DatasetCamVid
from argparse import ArgumentParser
import torch
from torch.utils.data import  DataLoader
from pathlib import Path
from train import Trainer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from model.BLNet import Net
from config import Config
from loss import CrossEntropyLoss2d, LovaszSoftmax
import numpy as np

def main(args):
    if args.dataset == "cityscapes":
        train_dataset = DatasetTrain(cityscapes_data_path="/home/chenxiaoshuang/Cityscapes",
                                    cityscapes_meta_path="/home/chenxiaoshuang/Cityscapes/gtFine", 
                                    only_encode=args.only_encode, extra_data=args.extra_data)
        val_dataset = DatasetVal(cityscapes_data_path="/home/chenxiaoshuang/Cityscapes",
                                cityscapes_meta_path="/home/chenxiaoshuang/Cityscapes/gtFine",
                                only_encode=args.only_encode)
        test_dataset = DatasetTest(cityscapes_data_path="/home/chenxiaoshuang/Cityscapes",
                                cityscapes_meta_path="/home/chenxiaoshuang/Cityscapes/gtFine")      
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size, shuffle=False, num_workers=8)
        num_classes = 20
    elif args.dataset == "camvid":
        train_dataset = DatasetCamVid(camvid_data_path="/home/chenxiaoshuang/CamVid",
                                    camvid_meta_path="/home/chenxiaoshuang/CamVid",
                                    only_encode=args.only_encode, mode="train")
        val_dataset = DatasetCamVid(camvid_data_path="/home/chenxiaoshuang/CamVid",
                                    camvid_meta_path="/home/chenxiaoshuang/CamVid",
                                    only_encode=args.only_encode, mode="val")
        test_dataset = DatasetCamVid(camvid_data_path="/home/chenxiaoshuang/CamVid",
                                    camvid_meta_path="/home/chenxiaoshuang/CamVid",
                                    only_encode=args.only_encode, mode="test")
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size, shuffle=False, num_workers=8)
        num_classes = 12
    else:
        print("Unsupported Dataset!")
        return

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    device_ids = [args.cuda, args.cuda+1]
    cfg=Config(args.dataset, args.only_encode, args.extra_data)
    net = Net(num_classes=num_classes)
    
    if torch.cuda.is_available():
        weight = cfg.weight.to(device)
    criterion1 = CrossEntropyLoss2d(weight)
    criterion2 = LovaszSoftmax(weight=weight)
    
    optimizer = optim.Adam(net.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    lambda1 = lambda epoch : (1 - epoch/300) ** 0.9

    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)
    
    trainer = Trainer('training', optimizer, exp_lr_scheduler, net, cfg, './log', device, device_ids, num_classes)
    trainer.load_weights(trainer.find_last(), encode=False, restart=False)
    #trainer.train(train_loader, val_loader, criterion1, criterion2, 300)
    trainer.evaluate(val_loader)
    trainer.test(test_loader)
    
    print('Finished Training')

if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--only-encode', action='store_true', default=False)
    parser.add_argument('--extra-data', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default="cityscapes")
    args = parser.parse_args()
    main(args)
