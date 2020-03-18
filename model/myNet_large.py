'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.functional import interpolate as interpolate

class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=True):
        super(Conv2dBnRelu,self).__init__()
		
        self.conv = nn.Sequential(
		nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
		nn.BatchNorm2d(out_ch, eps=1e-3),
		nn.ReLU(inplace=True)
	)

    def forward(self, x):
        return self.conv(x)

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = nn.Conv2d(160, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.hs2 = hswish()
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=False)

    def forward(self, x, predict=False):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        if predict:
            out = self.output_conv(out)
        return out

class APN_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        # global pooling branch
        self.branch1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
	)
        # midddle branch
        self.mid = nn.Sequential(
		Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
	)
        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)
		
        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)
		
        self.down3 = nn.Sequential(
		Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
		Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1)
	)
		
        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)
	
    def forward(self, x):
        
        h = x.size()[2]
        w = x.size()[3]
        
        b1 = self.branch1(x)
        b1= interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)
	
        mid = self.mid(x)
		
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3= interpolate(x3, size=(h // 4, w // 4), mode="bilinear", align_corners=True)	
        x2 = self.conv2(x2)
        x = x2 + x3
        x= interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
       		
        x1 = self.conv1(x1)
        x = x + x1
        x= interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        		
        x = torch.mul(x, mid)

        x = x + b1
       
       
        return x
          

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.apn = APN_Module(in_ch=128,out_ch=num_classes)
  
    def forward(self, input):
        
        output = self.apn(input)
        out = interpolate(output, size=(512, 1024), mode="bilinear", align_corners=True)
        return out
        
class myNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  
        super().__init__()

        if (encoder == None):
            self.encoder = MobileNetV3_Small(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    
            return self.decoder.forward(output)
            


def test():
    net = myNet(20)
    net = net.eval()
    x = torch.randn(1,3,512,1024)
    y = net(x)
    print(y.size())
    
if __name__ == "__main__":
    test()