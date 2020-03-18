import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2

def channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize,groups,
        channels_per_group,height,width)
    
    x = torch.transpose(x,1,2).contiguous()
    
    # flatten
    x = x.view(batchsize,-1,height,width)
    
    return x
    

class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=False):
        super().__init__()
		
        self.conv = nn.Sequential(
		nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
		#nn.BatchNorm2d(out_ch),
		nn.ReLU(inplace=True)
	)

    def forward(self, x):
        return self.conv(x)


# after Concat -> BN, you also can use Dropout like SS_nbt_module may be make a good result!
class DownsamplerBlock (nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel-in_channel, (3, 3), stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        #self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        #output = self.bn(output)
        output = self.relu(output)
        return output


class SS_nbt_module(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        oup_inc = chann//2
        
        # dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=False)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=False)

        #self.bn1_l = nn.BatchNorm2d(oup_inc)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=False, dilation = (dilated,1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=False, dilation = (1,dilated))

        #self.bn2_l = nn.BatchNorm2d(oup_inc)
        
        # dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=False)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=False)

        #self.bn1_r = nn.BatchNorm2d(oup_inc)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=False, dilation = (dilated,1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=False, dilation = (1,dilated))

        #self.bn2_r = nn.BatchNorm2d(oup_inc)       
        
        self.relu = nn.ReLU(inplace=True)
        #self.dropout = nn.Dropout2d(dropprob)       
        
    @staticmethod
    def _concat(x,out):
        return torch.cat((x,out),1)    
    
    def forward(self, input):

        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        #output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        #output1 = self.bn2_l(output1)
    
    
        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        #output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        #output2 = self.bn2_r(output2)

        #if (self.dropout.p != 0):
        #    output1 = self.dropout(output1)
        #    output2 = self.dropout(output2)

        out = self._concat(output1,output2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out,2)



class Context_path(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_block = DownsamplerBlock(3,32)

        self.layers = nn.ModuleList()

        for x in range(0, 3):
            self.layers.append(SS_nbt_module(32, 0.03, 1))
        

        self.layers.append(DownsamplerBlock(32,64))
        

        for x in range(0, 2):
            self.layers.append(SS_nbt_module(64, 0.03, 1))
  
        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 1):    
            self.layers.append(SS_nbt_module(128, 0.3, 1))
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
        
        self.layers2 = nn.ModuleList()
            
        for x in range(0, 1):    
            self.layers2.append(SS_nbt_module(128, 0.3, 2))
            self.layers2.append(SS_nbt_module(128, 0.3, 5))
            self.layers2.append(SS_nbt_module(128, 0.3, 9))
            self.layers2.append(SS_nbt_module(128, 0.3, 17))
                    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, input, predict=False):
        
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
            
        output1 = output
        
        for layer in self.layers2:
            output = layer(output)
            
        output2 = output
        
        tail = self.avg_pool(output)
        tail = interpolate(tail, size=output.size()[2:], mode='bilinear', align_corners=True)

        return output1, output2, tail
        
class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        #self.bn = nn.BatchNorm2d(out_ch)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        x = self.avgpool(input)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = Conv2dBnRelu(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class Spatial_path(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        inner_ch = 64
        self.conv_7x7 = Conv2dBnRelu(in_ch, inner_ch, kernel_size=7, stride=2, padding=3)
        self.conv_3x3_1 = Conv2dBnRelu(inner_ch, inner_ch, kernel_size=3, stride=2, padding=1)
        self.conv_3x3_2 = Conv2dBnRelu(inner_ch, inner_ch, kernel_size=3, stride=2, padding=1)
        self.conv_1x1 = Conv2dBnRelu(inner_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.context_path = Context_path()
        self.attention_refinement_module1 = AttentionRefinementModule(128, 128)
        self.attention_refinement_module2 = AttentionRefinementModule(128, 128)
        self.feature_fusion_module = FeatureFusionModule(256, 128)
        self.spatial_path = Spatial_path(3, 128)
        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=False)

    def forward(self, input, predict=False):
        spatial_out = self.spatial_path(input)
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        context_out = cx1 + cx2 + tail
        out = self.feature_fusion_module(spatial_out, context_out)

        if predict:
            out = self.output_conv(out)
        return out

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=2, padding=1)#nn.MaxPool2d(2)

    def forward(self, input):
        x = self.pool(input)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out
        out = interpolate(out, size=input.size()[2:], mode="bilinear", align_corners=True)
        out = out + input
        return out


class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.pool = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=2, padding=1)#nn.MaxPool2d(2)

    def forward(self,input):
        x = self.pool(input)
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out
        out = interpolate(out, size=input.size()[2:], mode="bilinear", align_corners=True)
        out = out + input
        return out 

class Pixel_shuffle(nn.Module):
    def __init__(self, inplanes, scale, num_class=20, pad=0):
        super().__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class*scale*scale, kernel_size=1, padding=pad, bias=False)
        self.ps = nn.PixelShuffle(scale)
    
    def forward(self, x):
        x = self.conv_w(x)
        x = self.ps(x)
        
        return x



class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #self.pam = PAM_Module(128)
        #self.cam = CAM_Module(128)
        self.apn = Pixel_shuffle(128, 8)
  
    def forward(self, input):
        #out = self.pam(input) + self.cam(input)
        out = self.apn(input)
        #out = interpolate(out, size=(512, 1024), mode="bilinear", align_corners=True)
        return out


class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    
            return self.decoder.forward(output)
