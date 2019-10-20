import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class plainEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(plainEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
    
class plainDecoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(plainDecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)
        
        self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        
        self.up = None
        
        if stride != 1:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if self.up is not None:
            x = self.up(x)
        
        return x
    

class resEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(resEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        
        self.downsample = None
        if stride != 1:  
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outChannel))
        
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out
    
class resDecoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(resDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)
        
        self.downsample = None
        self.up = None
        
        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            #self.up = nn.Upsample(scale_factor=2, mode='nearest')
            
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=1),
                nn.BatchNorm2d(outChannel),
                nn.Upsample(scale_factor=2, mode='bilinear'))
                #nn.Upsample(scale_factor=2, mode='nearest'))   
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.up is not None:
            out = self.up(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out
    
    
class GeoNet(nn.Module):
    def __init__(self, layers):
        super(GeoNet, self).__init__()  
        
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        
        self.en_layer1 = self.make_encoder_layer(plainEncoderBlock, 64, 64, layers[0], stride=1)  
        self.en_layer2 = self.make_encoder_layer(resEncoderBlock, 64, 128, layers[1], stride=2)
        self.en_layer3 = self.make_encoder_layer(resEncoderBlock, 128, 256, layers[2], stride=2)
        self.en_layer4 = self.make_encoder_layer(resEncoderBlock, 256, 512, layers[3], stride=2)
        self.en_layer5 = self.make_encoder_layer(resEncoderBlock, 512, 512, layers[4], stride=2)
        
        self.de_layer5 = self.make_decoder_layer(resDecoderBlock, 512, 512, layers[4], stride=2)
        self.de_layer4 = self.make_decoder_layer(resDecoderBlock, 512, 256, layers[3], stride=2)
        self.de_layer3 = self.make_decoder_layer(resDecoderBlock, 256, 128, layers[2], stride=2)
        self.de_layer2 = self.make_decoder_layer(resDecoderBlock, 128, 64, layers[1], stride=2)
        self.de_layer1 = self.make_decoder_layer(plainDecoderBlock, 64, 64, layers[0], stride=1)
        
        self.conv_end = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        
        self.fconv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fbn = nn.BatchNorm2d(64)
        self.f_en_layer1 = self.make_encoder_layer(plainEncoderBlock, 64, 64, layers[0], stride=1)  
        self.f_en_layer2 = self.make_encoder_layer(resEncoderBlock, 64, 128, layers[1], stride=2)
        self.f_en_layer3 = self.make_encoder_layer(resEncoderBlock, 128, 256, layers[2], stride=2)
        self.f_en_layer4 = self.make_encoder_layer(resEncoderBlock, 256, 512, layers[3], stride=2)
        self.f_en_layer5 = self.make_encoder_layer(resEncoderBlock, 512, 512, layers[4], stride=2)
        
        self.f_en_layer6 = self.make_encoder_layer(resEncoderBlock, 512, 512, 1, stride=2)
        self.f_en_layer7 = self.make_encoder_layer(resEncoderBlock, 512, 512, 1, stride=2)
        self.f_en_layer8 = self.make_encoder_layer(resEncoderBlock, 512, 512, 1, stride=2)
        
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc1bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2bn = nn.BatchNorm1d(512)
        
        self.catconv = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.catbn  = nn.BatchNorm2d(512)
        
                       
        # weight initializaion with Kaiming method
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def make_encoder_layer(self, block, inChannel, outChannel, block_num, stride):

        layers = []
        layers.append(block(inChannel, outChannel, stride=stride))
        for i in range(1, block_num):
            layers.append(block(outChannel, outChannel, stride=1))

        return nn.Sequential(*layers)
    
    def make_decoder_layer(self, block, inChannel, outChannel, block_num, stride):

        layers = []
        for i in range(0, block_num-1):
            layers.append(block(inChannel, inChannel, stride=1))
            
        layers.append(block(inChannel, outChannel, stride=stride))
        
        return nn.Sequential(*layers)
                       
    def forward(self, x, y):
        
        x = F.relu(self.bn(self.conv(x)))
        x = self.en_layer1(x)     #256
        x = self.en_layer2(x)     #128
        x = self.en_layer3(x)     #64
        x = self.en_layer4(x)     #32
        x = self.en_layer5(x)     #16
        
        y = F.relu(self.fbn(self.fconv(y)))
        y = self.f_en_layer1(y)     #256
        y = self.f_en_layer2(y)     #128
        y = self.f_en_layer3(y)     #64
        y = self.f_en_layer4(y)     #32
        y = self.f_en_layer5(y)     #16
        
        y = self.f_en_layer6(y)     #8
        y = self.f_en_layer7(y)     #4
        y = self.f_en_layer8(y)     #2
        
        y = y.view(-1, 512*2*2)
        
        y = F.relu(self.fc1bn(self.fc1(y)))
        y = F.relu(self.fc2bn(self.fc2(y)))
        y = y.unsqueeze(2).unsqueeze(2).expand_as(x)
        
        x = torch.cat([x, y], dim=1)
        
        x = F.relu(self.catbn(self.catconv(x)))
        
        x = self.de_layer5(x)     
        x = self.de_layer4(x)     
        x = self.de_layer3(x)     
        x = self.de_layer2(x)     
        x = self.de_layer1(x)        
        
        x = self.conv_end(x)
        return x
        
class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output - target + 1e-16, p=2, dim=1).mean()
        return lossvalue
        