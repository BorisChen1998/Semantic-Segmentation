from argparse import ArgumentParser
import torch
from torch.utils.data import  DataLoader
from pathlib import Path
import os
from train import Trainer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from model.BLNet import Net
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage, transforms
import numpy as np
from PIL import Image, ImageOps
dic = torch.tensor([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 255])
color_dic = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]])
class DatasetW(torch.utils.data.Dataset):
    def __init__(self, camvid_data_path, camvid_meta_path, only_encode, mode):
        self.img_dir = camvid_data_path + "/{}/".format(mode)
        self.label_dir = camvid_meta_path + "/{}annot/".format(mode)
        self.only_encode = only_encode
        self.mode = mode
        self.img_h = 720
        self.img_w = 960

        self.new_img_h = 720
        self.new_img_w = 960

        self.examples = []
        

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_path = self.img_dir + file_name

            label_img_path = self.label_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = file_name
            self.examples.append(example)

        self.num_examples = len(self.examples)
        self.tran=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        label_img_path = example["label_img_path"]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        with open(label_img_path, "rb") as f:
            label_img = Image.open(f).convert("P")
            
        img =  Resize(self.new_img_h, Image.BILINEAR)(img)
        label_img = Resize(self.new_img_h, Image.NEAREST)(label_img)

        img = self.tran(img)
        if self.only_encode:
            label_img = Resize(int(self.new_img_h/8), Image.NEAREST)(label_img)
        
        label_img = torch.from_numpy(np.array(label_img)).long()
        label_img[label_img == 255] = num_classes-1
        return (img, label_img)

    def __len__(self):
        return self.num_examples
num_classes = 12
torch.cuda.set_device(2)
normVal = 1.1
if __name__=='__main__':
    
    dataset = DatasetW(camvid_data_path="/home/chenxiaoshuang/CamVid",
                                    camvid_meta_path="/home/chenxiaoshuang/CamVid",
                                    only_encode=False, mode="train")
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)
    num = torch.tensor([0 for i in range(num_classes)]).cuda()
    res = torch.tensor([0.0 for i in range(num_classes)]).cuda()
    for data in loader:
        inputs, labels = data
        labels = labels.cuda()
        for c in range(num_classes):
            num[c] += (labels == c).sum()
    
    num = num * 1.0 / num.sum()
    print(num)
    for i in range(num_classes):
        res[i] = 1 / torch.log(normVal + num[i])
    print(res)