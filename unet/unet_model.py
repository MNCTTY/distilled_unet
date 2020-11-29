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
        
        self.bottleneck1_1 = branchBottleNeck(64 , self.num_classes, kernel_size=32) ## вот вопрос только, нужно ли нам тут self.num_classes или какая-то другая интересная размерность? По идее пока каждый класс в отдельном слое предсказывается и должен быть заполнен вероятностным распределением нахождения каждого класса на пикселе
        
        ## еще один вопрос про kernel size здесь
        
        self.avgpool1 = nn.AdaptiveAvgPool2d((600,600)) # was (1,1)
        self.middle_fc1 = nn.Linear(512 * 4, self.num_classes) ##### какое количество входных фичей?
        self.soft1 = nn.Softmax(dim=1)
        
        self.bottleneck2_1 = branchBottleNeck(128 , self.num_classes , kernel_size=16)
        self.avgpool2 = nn.AdaptiveAvgPool2d((600,600)) # was (1,1)
#         self.middle_fc2 = nn.Linear(512 * block.expansion, self.num_classes)
        self.soft2 = nn.Softmax(dim=1)
        
        self.bottleneck3_1 = branchBottleNeck(256 , self.num_classes , kernel_size=8)
        self.avgpool3 = nn.AdaptiveAvgPool2d((600,600)) # was (1,1)
#         self.middle_fc3 = nn.Linear(512 * block.expansion, self.num_classes)
        self.soft3 = nn.Softmax(dim=1)
        
        self.bottleneck4_1 = branchBottleNeck(512 , self.num_classes , kernel_size=4)
        self.avgpool4 = nn.AdaptiveAvgPool2d((600,600)) # was (1,1)
#         self.middle_fc4 = nn.Linear(512 * block.expansion, self.num_classes)
        self.soft4 = nn.Softmax(dim=1)

        self.bottleneck5_1 = branchBottleNeck(1024 , self.num_classes , kernel_size=2)
        self.avgpool5 = nn.AdaptiveAvgPool2d((600,600)) # was (1,1)
#         self.middle_fc5 = nn.Linear(512 * block.expansion, self.num_classes)
        self.soft5 = nn.Softmax(dim=1)
        
        self.soft_fin = nn.Softmax(dim=1)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        x1 = self.inc(x) # main flow
        
        btn1 = self.bottleneck1_1(x1)
        btn1 = self.avgpool1(btn1)
        btn1 = self.middle_fc1(btn1)
#         btn1 = self.up4(x1,btn1)
        soft_out1 = self.soft5(btn1)
        
        x2 = self.down1(x1) # main flow
        
        btn2 = self.bottleneck2_1(x2)
        btn2 = self.avgpool2(btn2)
#         btn2 = self.up3(x2,bt2)
        soft_out2 = self.soft2(btn2)
        
        x3 = self.down2(x2) # main flow
        
        btn3 = self.bottleneck3_1(x3)
        btn3 = self.avgpool3(btn3)
#         btn3 = self.up2(x3,bt3)
        soft_out3 = self.soft3(btn3)
        
        x4 = self.down3(x3) # main flow
        
        btn4 = self.bottleneck4_1(x4)
        btn4 = self.avgpool4(btn4)
#         btn4 = self.up1(x4,bt4)
        soft_out4 = self.soft4(btn4)

        x5 = self.down4(x4) # main flow
        
        x = self.up1(x5, x4) # main flow
        x = self.up2(x, x3) # main flow
        x = self.up3(x, x2) # main flow
        x = self.up4(x, x1) # main flow
        
        soft_fin = self.soft_fin(x)
        
        logits = self.outc(soft_fin) # main flow # was x
                
        print('outputs:')
        print(logits.shape)
        print(soft_fin.shape)
        print(btn1.shape)
        print(btn2.shape)
        print(btn3.shape)
        print(btn4.shape)
        print(soft_out1.shape)

        return logits, soft_fin, btn1, btn2, btn3, btn4, soft_out1, soft_out2, soft_out3, soft_out4

    