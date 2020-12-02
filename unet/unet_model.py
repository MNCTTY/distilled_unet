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
        
        self.avg_dim = 600 # was 1
        self.fc_out_dim = 600
        
        self.bottleneck1_1 = branchBottleNeck(64 , self.num_classes, kernel_size=32) 
                
        self.avgpool1 = nn.AdaptiveAvgPool2d((self.avg_dim,self.avg_dim)) 
        self.middle_fc1 = nn.Linear(self.fc_out_dim, self.fc_out_dim)
        self.up_ord1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.soft1 = nn.Softmax(dim=1)
        
        self.bottleneck2_1 = branchBottleNeck(128 , self.num_classes , kernel_size=16)
        self.avgpool2 = nn.AdaptiveAvgPool2d((self.avg_dim,self.avg_dim)) 
        self.middle_fc2 = nn.Linear(self.avg_dim, self.fc_out_dim)
        self.soft2 = nn.Softmax(dim=1)
        
        self.bottleneck3_1 = branchBottleNeck(256 , self.num_classes , kernel_size=8)
        self.avgpool3 = nn.AdaptiveAvgPool2d((self.avg_dim,self.avg_dim)) 
        self.middle_fc3 = nn.Linear(self.avg_dim, self.fc_out_dim)
        self.soft3 = nn.Softmax(dim=1)
        
        self.bottleneck4_1 = branchBottleNeck(512 , self.num_classes , kernel_size=4)
        self.avgpool4 = nn.AdaptiveAvgPool2d((self.avg_dim,self.avg_dim)) 
        self.middle_fc4 = nn.Linear(self.avg_dim, self.fc_out_dim)
        self.soft4 = nn.Softmax(dim=1)

        self.bottleneck5_1 = branchBottleNeck(1024 , self.num_classes , kernel_size=2)
        self.avgpool5 = nn.AdaptiveAvgPool2d((self.avg_dim,self.avg_dim)) 
        self.middle_fc5 = nn.Linear(self.avg_dim, self.fc_out_dim)
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
#         making copies to be sure that main flow doesnt be affected by new transforms
#         y = x.clone().detach() or y = torch.tensor(x)


        x1_copy = x1.clone().detach()
        btn1 = self.bottleneck1_1(x1_copy)
        btn1 = self.avgpool1(btn1)
        btn1 = self.middle_fc1(btn1)
        soft_out1 = self.soft5(btn1)
        
        x2 = self.down1(x1) # main flow
        
        x2_copy = x2.clone().detach()
        btn2 = self.bottleneck2_1(x2_copy)
        btn2 = self.avgpool2(btn2)
        btn2 = self.middle_fc1(btn2)
        soft_out2 = self.soft2(btn2)
        
        x3 = self.down2(x2) # main flow
        
        x3_copy = x3.clone().detach()
        btn3 = self.bottleneck3_1(x3_copy)
        btn3 = self.avgpool3(btn3)
        btn3 = self.middle_fc1(btn3)
        soft_out3 = self.soft3(btn3)
        
        x4 = self.down3(x3) # main flow
        
        x4_copy = x4.clone().detach()
        btn4 = self.bottleneck4_1(x4_copy)
        btn4 = self.avgpool4(btn4)
        btn4 = self.middle_fc1(btn4)
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

    