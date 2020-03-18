import os
import numpy as np
import torch
import time
from  cityscape  import DatasetTrain ,DatasetVal, DatasetTest
from torch.utils.data import  DataLoader
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable

from model.BLNet_no_bn import Net

#import torch.backends.cudnn as cudnn
#cudnn.benchmark = True
def main(args):
    torch.cuda.set_device(args.cuda)
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    model = Net(20)
    model = model.to(device)#.half()	#HALF seems to be doing slower for some reason
    #model = torch.nn.DataParallel(model).cuda()
    model.eval()
    images = torch.randn(args.batch_size, args.num_channels, args.height, args.width)
    images = images.to(device)

    time_train = []

    i=0
    with torch.no_grad():
        while(True):
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            fwt = time.time() - start_time
            if i>2:
                time_train.append(fwt)
                print ("Forward time per img (b=%d): %.5f (Mean: %.5f)" % (args.batch_size, fwt/args.batch_size, sum(time_train) / len(time_train) / args.batch_size))
            time.sleep(1)
            i+=1

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--cuda', type=int, default=0)
    #parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())