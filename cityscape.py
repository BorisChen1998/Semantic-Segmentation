import torch
import torch.utils.data
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage, transforms
import numpy as np
from PIL import Image, ImageOps
import os
import random

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]

val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin/", "bielefeld/", "bonn/", "leverkusen/", "mainz/", "munich/"]
num_classes = 20

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, only_encode):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/train/"
        self.label_dir = cityscapes_meta_path + "/train/"
        self.only_encode = only_encode
        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for train_dir in train_dirs:
            train_img_dir_path = self.img_dir + train_dir
            label_img_dir_path = self.label_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = label_img_dir_path + img_id + "_gtFine_labelTrainIds.png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
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
        label_img[label_img == 255] = 19
        return (img, label_img)

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, only_encode):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/val/"
        self.label_dir = cityscapes_meta_path + "/val/"
        self.only_encode = only_encode
        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for val_dir in val_dirs:
            val_img_dir_path = self.img_dir + val_dir
            label_img_dir_path = self.label_dir + val_dir

            file_names = os.listdir(val_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = val_img_dir_path + file_name

                label_img_path = label_img_dir_path + img_id + "_gtFine_labelTrainIds.png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)
        self.tran=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

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
        label_img[label_img == 255] = 19
        return (img, label_img)

    def __len__(self):
        return self.num_examples
        
        
        
class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/test/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for test_dir in test_dirs:
            test_img_dir_path = self.img_dir + test_dir

            file_names = os.listdir(test_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = test_img_dir_path + file_name

                example = {}
                example["img_path"] = img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"] 
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            
        img =  Resize(self.new_img_h, Image.BILINEAR)(img)

        img = ToTensor()(img)

        return (img, img_id)

    def __len__(self):
        return self.num_examples


if __name__=='__main__':
    train_dataset = DatasetTrain(cityscapes_data_path="/home/shen/Data/DataSet/Cityscape",
                                cityscapes_meta_path="/home/shen/Data/DataSet/Cityscape/gtFine/")
    val_dataset = DatasetVal(cityscapes_data_path="/home/shen/Data/DataSet/Cityscape",
                             cityscapes_meta_path="/home/shen/Data/DataSet/Cityscape/gtFine")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1, shuffle=True,
                                           num_workers=1)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=8, shuffle=False,
                                         num_workers=1)

    # for img,lab in train_loader:
    #     print(img.size(),lab.size())

    img,labs=train_dataset[0]
    print(labs.size())

    img,labs=val_dataset[0]
    print(img.size())
    print(labs.size())     
