import torch
import torch.utils.data
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage, transforms
import numpy as np
from PIL import Image, ImageOps
import os
import random

num_classes = 12

class DatasetCamVid(torch.utils.data.Dataset):
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
        
        if self.mode == "train":    
            hflip = random.random()
            if (hflip < 0.5):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
                
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            img = ImageOps.expand(img, border=(transX,transY,0,0), fill=0)
            label_img = ImageOps.expand(label_img, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            img = img.crop((0, 0, img.size[0]-transX, img.size[1]-transY))
            label_img = label_img.crop((0, 0, label_img.size[0]-transX, label_img.size[1]-transY))   

        img = self.tran(img)
        if self.only_encode:
            label_img = Resize(int(self.new_img_h/8), Image.NEAREST)(label_img)
        
        label_img = torch.from_numpy(np.array(label_img)).long()
        label_img[label_img == 255] = num_classes-1
        return (img, label_img)

    def __len__(self):
        return self.num_examples