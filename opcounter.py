from thop import profile
import torch
from model.BLNet_no_bn import Net

net=Net(num_classes=20)

input = torch.randn(1,3,512,1024)

flops, params = profile(net, inputs=(input, ))
print("params:",params,"flops:",flops)
