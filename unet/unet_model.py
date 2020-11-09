""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import torch.nn as nn

 
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear 

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        self.dim = 1
        
        
        self.num_classes = n_classes
        
        self.bottleneck1_1 = branchBottleNeck(64 , self.num_classes, kernel_size=32)
        
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
#         self.middle_fc1 = nn.Linear(self.num_classes, num_classes)
#         self.soft1 = Softmax_layer(dim = 1)
        self.soft1 = nn.Softmax(dim=1)
        
        self.bottleneck2_1 = branchBottleNeck(128 , self.num_classes , kernel_size=16)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
#         self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)
#         self.soft2 = Softmax_layer(dim=1)
        self.soft2 = nn.Softmax(dim=1)
        
        self.bottleneck3_1 = branchBottleNeck(256 , self.num_classes , kernel_size=8)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
#         self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)
#         self.soft3 = Softmax_layer(dim=1)
        self.soft3 = nn.Softmax(dim=1)
        
        self.bottleneck4_1 = branchBottleNeck(512 , self.num_classes , kernel_size=4)
        self.avgpool4 = nn.AdaptiveAvgPool2d((1,1))
#         self.middle_fc4 = nn.Linear(512 * block.expansion, num_classes)
#         self.soft4 = Softmax_layer(dim=1)
        self.soft4 = nn.Softmax(dim=1)

        self.bottleneck5_1 = branchBottleNeck(1024 , self.num_classes , kernel_size=2)
        self.avgpool5 = nn.AdaptiveAvgPool2d((1,1))
#         self.soft5 = Softmax_layer(dim=1)
        self.soft5 = nn.Softmax(dim=1)
        
#         self.soft_fin = Softmax_layer(dim=1)
        self.soft_fin = nn.Softmax(dim=1)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, layers, stride=1):
        
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer = []
        layer.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            layer.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layer)

    def forward(self, x):
        x1 = self.inc(x)
        btn1 = self.bottleneck1_1(x1)
        btn1 = self.avgpool1(btn1)
        soft_out1 = self.soft5(btn1)
        
        x2 = self.down1(x1)
        btn2 = self.bottleneck2_1(x2)
        btn2 = self.avgpool2(btn2)
        soft_out2 = self.soft2(btn2)
        
        x3 = self.down2(x2)
        btn3 = self.bottleneck3_1(x3)
        btn3 = self.avgpool3(btn3)
        soft_out3 = self.soft3(btn3)
        
        
        x4 = self.down3(x3)
        btn4 = self.bottleneck4_1(x4)
        btn4 = self.avgpool4(btn4)
        soft_out4 = self.soft4(btn4)

        x5 = self.down4(x4)
#         btn5 = self.bottleneck5_1(x5)
#         btn5 = self.avgpool5(btn5)
#         soft_out5 = self.soft5(btn5)

        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.soft_fin(x)
        
        logits = self.outc(x)
        
#         btn5,
        
        return logits, btn1, btn2, btn3, btn4, soft_out1, soft_out2, soft_out3, soft_out4 #, soft_out5

    