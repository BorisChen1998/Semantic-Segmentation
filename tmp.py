import argparse
import torch
import torchvision
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage, transforms
from pathlib import Path
import yaml
import torch.nn as nn
from model.BLNet import Net
from config import Config
import numpy as np
import onnx
from PIL import Image, ImageOps
import cv2
dic = torch.tensor([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 255])
color_dic = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]])
if __name__=='__main__':
    #net = Net(num_classes=20)
    #checkpoint = torch.load('./log/mynet20200305T1326/model_mynet_0275.pt')
    #net.load_state_dict(checkpoint['model_state_dict'])
    #net.eval()
    with open("./test2.png", "rb") as f:
        img = Image.open(f).convert("RGB")
    img =  Resize(512, Image.BILINEAR)(img)
    #img = ToTensor()(img)
    #img = img.unsqueeze(0)
    #preds = net(img)
    #preds = preds.detach()
    #pred = preds[0,:,:,:]
    #pred = torch.argmax(pred, axis=0)
    #pred = color_dic[pred]
    #pred = pred.numpy()
    img.save("./result2.png")
    #cv2.imwrite("./result2.png", pred)
    '''
    img = torch.rand(1, 3, 512, 1024)
    output_onnx = 'BLNet.onnx'
    print("Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    output_names = ["output"]
    torch_out = torch.onnx._export(net, img, output_onnx, 
        export_params=True, 
        verbose=False, 
        input_names=input_names, 
        output_names=output_names,
        opset_version=11
    )


    print("Loading and checking exported model from '{}'".format(output_onnx))
    onnx_model = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_model)
    print("Passed")
    '''
