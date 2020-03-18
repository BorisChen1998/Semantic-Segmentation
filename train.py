import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os
import datetime
from tensorboardX import SummaryWriter
import time
import sys, time
import torch.nn.functional as F
from utils.metrics import compute_iou_batch
import numpy as np
from argparse import ArgumentParser
import cv2

class Trainer(object):
    def __init__(self, mode, optim, scheduler, model, config, model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.model = model
        self.cuda = torch.cuda.is_available()
        self.model_dir = model_dir
        self.optim = optim
        self.epoch = 0
        self.num_classes = 20
        self.config = config
        self.scheduler = scheduler
        self.set_log_dir()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        #device_ids = [0, 1]
        #self.model = nn.DataParallel(self.model, device_ids)
        
    def train(self, train_loader, val_loader, loss_function, num_epochs):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataloaders = {'train': train_loader, 'val': val_loader}

        writer = SummaryWriter(log_dir=self.log_dir)

        for epoch in range(self.epoch, num_epochs):
            since = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    print("lr: ", self.optim.param_groups[0]['lr'])
                else:
                    self.model.eval()
                running_loss = 0.0
                bar_steps = len(dataloaders[phase])
                #process_bar = ShowProcess(bar_steps)
                total = 0

                nums = torch.tensor([0 for i in range(self.num_classes-1)]).cuda()
                dens = torch.tensor([0 for i in range(self.num_classes-1)]).cuda()
                
                for i, data in enumerate(dataloaders[phase], 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    self.optim.zero_grad()
                    #forward
                    #track history if only in train
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = loss_function(outputs, labels)
                        #preds = F.interpolate(outputs[0], size=labels.size()[2:], mode='bilinear', align_corners=True)
                        preds_np=outputs.detach()
                        labels_np = labels.detach()

                        num, den = compute_iou_batch(preds_np, labels_np)
                        nums += num
                        dens += den
                        #backward+optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optim.step()
                
                
                    # statistics
                    total += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    #process_bar.show_process()
                #process_bar.close()
                if phase == 'train':
                    #self.scheduler.step(loss.cpu().data.numpy())
                    if self.scheduler:
                        self.scheduler.step()
                epoch_loss = running_loss / total
                ious = nums*1.0 / dens
                iou=torch.mean(ious).item()
                print('{} Loss: {:.4f} iou:{:.4f} '.format(phase, epoch_loss,iou))
                
                writer.add_scalar('{}_loss'.format(phase), epoch_loss, epoch)
                writer.add_scalar('{}_iou'.format(phase),iou,epoch)
                

            time_elapsed = time.time() - since
            print('one epoch complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))


            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'lr_scheduler': self.scheduler.state_dict(),
                'loss': loss
            }, self.checkpoint_path.format(epoch))
        writer.close()
        print("train finished")

    def set_log_dir(self, model_path=None):
        if self.mode == 'training':
            now = datetime.datetime.now()
            #if we hanbe a model path with date and epochs use them
            if model_path:
                # Continue form we left of .Get epoch and date form the file name
                # A sample model path might look like:
                #/path/to/logs/coco2017.../DeFCN_0001.h5
                import re
                regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/model\_[\w-]+(\d{4})\.pt"
                m = re.match(regex, model_path)
                if m:
                    now = datetime.datetime(
                        int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        int(m.group(4)), int(m.group(5)))
                    # Epoch number in file is 1-based, and in Keras code it's 0-based.
                    # So, adjust for that then increment by one to start from the next epoch
                    #self.epoch = int(m.group(6))  + 1
                    print('Re-starting from epoch %d' % self.epoch)

                    # Directory for training logs
            self.log_dir = os.path.join(
                self.model_dir, "{}{:%Y%m%dT%H%M}".format(
                    self.config.NAME.lower(), now))
            
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.checkpoint_path = os.path.join(
                self.log_dir,
                "model_{}_*epoch*.pt".format(self.config.NAME.lower()))
            self.checkpoint_path = self.checkpoint_path.replace(
                "*epoch*", "{:04d}")

    def find_last(self,num=-1):
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)

        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find model directory under {}".format(
                    self.model_dir))
        
        if self.mode == 'training':
            dir_name = os.path.join(self.model_dir, dir_names[-2])
            print(dir_name)
            os.rmdir(os.path.join(self.model_dir, dir_names[-1]))
            
        else:
            dir_name = os.path.join(self.model_dir, dir_names[-1])
        
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("model"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[num])
        return checkpoint

    def load_weights(self, file_path, by_name=False, exclude=None):
        checkpoint = torch.load(file_path)
        self.model.encoder.load_state_dict(checkpoint['model_state_dict'])

        #self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        '''
        for param_group in self.optim.param_groups:
            param_group['lr'] = 1e-5
        lambda1 = lambda epoch : (1 - epoch/300) ** 0.9
        self.scheduler = lr_scheduler.LambdaLR(self.optim,lr_lambda=lambda1)
        '''
        
        #self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        checkpoint['epoch'] = 0
        self.epoch = checkpoint['epoch'] + 1
        #self.loss = checkpoint['loss']
        self.set_log_dir(file_path)
        print("load weights from {} finished.".format(file_path))

    def test(self, test_loader, save_dir="./result/"):
        #assert self.mode == "inference", "Create model in inference mode."
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dic = torch.tensor([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 255]).to(device)
        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, img_ids = data
                inputs = inputs.to(device)
                preds = self.model(inputs)
                preds = preds.detach()
                preds = F.interpolate(preds, size=(1024, 2048), mode='bilinear', align_corners=True)
                for pred, img_id in zip(preds, img_ids):
                    pred = torch.argmax(pred, axis=0)
                    pred = dic[pred]
                    pred = pred.cpu().numpy()
                    cv2.imwrite(save_dir+img_id+".png", pred)
            

    def evaluate(self, val_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model = self.model.module.to(device).eval()
        self.model = self.model.eval()
        

        with torch.no_grad():
            bar_steps = len(val_loader)
            nums = torch.tensor([0 for i in range(self.num_classes-1)]).cuda()
            dens = torch.tensor([0 for i in range(self.num_classes-1)]).cuda()
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                preds_np=outputs.detach()
                labels_np = labels.detach()

                num, den = compute_iou_batch(preds_np, labels_np)
                nums += num
                dens += den
            print(nums, dens)
            ious = nums*1.0 / dens
            print(ious)
            iou=torch.mean(ious).item()
            print('iou:{:.4f} '.format(iou))
            
    def fps(self, val_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model = self.model.module.to(device).eval()
        self.model = self.model.eval()
        
        with torch.no_grad():
            bar_steps = len(val_loader)
            res = []
            for data in val_loader:
                inputs = data[0].to(device)
                torch.cuda.synchronize()
                since = time.time()
                outputs = self.model(inputs)
                torch.cuda.synchronize()
                time_elapsed = time.time() - since
                print('{:.5f}s'.format(time_elapsed/val_loader.batch_size))
                res.append(time_elapsed)
                time.sleep(1)
                
                
            time_sum = 0
            for i in res:
                time_sum += i
            print('one epoch complete in {:.0f}m {:.2f}s'.format(
                time_sum // 60, time_sum % 60))
                
       
class ShowProcess():
    
    i = 0
    max_steps = 0
    max_arrow = 50

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0
        
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
            + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()

    def close(self, words='done'):
        print('')
        #print(words)
        self.i = 0


if __name__ == "__main__":
    """Here is an example to show how to impliement the class trainer."""
