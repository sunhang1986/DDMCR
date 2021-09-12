import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
# from spectral import SpectralNorm
import numpy as np
from soca_module import SOCA,MCC_module


class Generator_base(nn.Module):
    def __init__(self):
        super(Generator_base, self).__init__()
        self.nonlinear = nn.SELU(inplace=True)
        self.upsample_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_down1 = nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.conv_down2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv_down3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv_down4 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv_down5 = nn.Conv2d(128, 128, 5, stride=2, padding=2)

        self.conv_down6 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        self.conv_down7 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        #self.adaptmaxpool = nn.AdaptiveMaxPool2d((8,8))
        self.adaptmaxpool = nn.AdaptiveAvgPool2d((4,4))
        self.conv_soca = nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.conv_at = nn.Conv2d(256, 256, 1, stride=1, padding=0)

        #self.conv_global1 = nn.Conv2d(128, 128, 8, stride=1, padding=0)
        self.conv_global1 = nn.Conv2d(128, 128, 4, stride=1, padding=0)
        self.conv_global2 = nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv_global3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv_up1 = nn.Conv2d(256, 128, 1, stride=1, padding=0)
        self.conv_up2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_up3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv_up4 = nn.Conv2d(192, 64, 3, stride=1, padding=1)
        self.conv_up5 = nn.Conv2d(96, 32, 3, stride=1, padding=1)

        """
        self.conv_up2 = nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0)
        self.conv_up3 = nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0)
        self.conv_up4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.conv_up5 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0)
        """

        self.conv_up6 = nn.Conv2d(48, 16, 3, stride=1, padding=1)
        self.conv_up7 = nn.Conv2d(16, 3, 5, stride=1, padding=2)

class BNIN(nn.Module):
    def __init__(self, planes):
        super(BNIN, self).__init__()
        half1 = int(planes*0.6)
        self.half = half1
        half2 = planes - half1
        self.BN = nn.BatchNorm2d(half1)
        self.IN = nn.InstanceNorm2d(half2, affine=True)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.BN(split[0].contiguous())
        out2 = self.IN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class Generator(nn.Module):
    def __init__(self, g_base):
        super(Generator, self).__init__()
        self.bn_down1 = nn.BatchNorm2d(16)
        self.bn_down2 = nn.BatchNorm2d(32)
        self.bn_down3 = nn.BatchNorm2d(64)
        self.bn_down4 = nn.BatchNorm2d(128)
        self.bn_down5 = nn.BatchNorm2d(128)
        self.bn_down6 = nn.BatchNorm2d(128)
        self.bn_down7 = nn.BatchNorm2d(128)
        self.bn_up1 = nn.BatchNorm2d(128)
        self.bn_up2 = nn.BatchNorm2d(256)
        self.bn_up3 = nn.BatchNorm2d(192)
        self.bn_up4 = nn.BatchNorm2d(96)
        self.bn_up5 = nn.BatchNorm2d(48)
        self.bn_up6 = nn.BatchNorm2d(16)

        self.soca128=SOCA(128)
        self.mcc128=MCC_module(128)
        #self.mcc128_5=MCC_module(128)
        self.self_attention=Self_Attention(128)

        #self.bn_down1 = nn.InstanceNorm2d(16)
        #self.bn_down2 = nn.InstanceNorm2d(32)
        #self.bn_down3 = nn.InstanceNorm2d(64)
        #self.bn_down4 = nn.InstanceNorm2d(128)
        #self.bn_down5 = nn.InstanceNorm2d(128)
        #self.bn_down6 = nn.InstanceNorm2d(128)
        #self.bn_down7 = nn.InstanceNorm2d(128)
        #self.bn_up1 = nn.InstanceNorm2d(128)
        #self.bn_up2 = nn.InstanceNorm2d(256)
        #self.bn_up3 = nn.InstanceNorm2d(192)
        #self.bn_up4 = nn.InstanceNorm2d(96)
        #self.bn_up5 = nn.InstanceNorm2d(48)
        #self.bn_up6 = nn.InstanceNorm2d(16)

         
        self.g = g_base
        self.sn_conv_down1 = self.g.conv_down1
        self.sn_conv_down2 = self.g.conv_down2
        self.sn_conv_down3 = self.g.conv_down3
        self.sn_conv_down4 = self.g.conv_down4
        self.sn_conv_down5 = self.g.conv_down5
        self.sn_conv_down6 = self.g.conv_down6
        self.sn_conv_down7 = self.g.conv_down7
        self.sn_conv_global1 = self.g.conv_global1
        self.sn_conv_global2 = self.g.conv_global2
        self.sn_conv_global3 = self.g.conv_global3
        self.sn_conv_up1 = self.g.conv_up1
        self.sn_conv_up2 = self.g.conv_up2
        self.sn_conv_up3 = self.g.conv_up3
        self.sn_conv_up4 = self.g.conv_up4
        self.sn_conv_up5 = self.g.conv_up5
        self.sn_conv_up6 = self.g.conv_up6

    def global_concat_layer(self, a, b):
        repeat_h = a.size(2)
        repeat_w = a.size(3)
        g_feature = b.repeat(1, 1, repeat_h, repeat_w)
        out = torch.cat((a, g_feature), 1)
        return out

    def concat_layer(self, a, b):
        out = torch.cat((a, b), 1)
        return out

    def forward(self, inputx):
        conv1 = self.sn_conv_down1(inputx)
        #conv1 = self.bn_down1_IN(conv1)
        l1 = self.bn_down1(self.g.nonlinear(conv1))
        conv2 = self.sn_conv_down2(l1)
        l2 = self.bn_down2(self.g.nonlinear(conv2))
        conv3 = self.sn_conv_down3(l2)
        l3 = self.bn_down3(self.g.nonlinear(conv3))
        conv4 = self.sn_conv_down4(l3)
        l4 = self.bn_down4(self.g.nonlinear(conv4))
        conv5 = self.sn_conv_down5(l4)
        l5 = self.bn_down5(self.g.nonlinear(conv5))
        #l5_mcc=self.mcc128_5(l5)
        conv6 = self.sn_conv_down6(l5)
        l6 = self.bn_down6(self.g.nonlinear(conv6))
        conv7 = self.sn_conv_down7(l6)
        l7 = self.bn_down7(self.g.nonlinear(conv7))

        l7_ = self.g.adaptmaxpool(l7)
        l8 = self.g.nonlinear(self.sn_conv_global1(l7_))
        l9 = self.sn_conv_global2(l8)
        l10 = self.global_concat_layer(self.sn_conv_global3(l5), l9)
        l11 = self.bn_up1(self.g.nonlinear(self.sn_conv_up1(l10)))
       
        l11_mcc = self.mcc128(l11)

        l12 = self.bn_up2(self.g.nonlinear(self.concat_layer(self.g.upsample_up(self.sn_conv_up2(l11_mcc)), conv4)))
        l13 = self.bn_up3(self.g.nonlinear(self.concat_layer(self.g.upsample_up(self.sn_conv_up3(l12)), conv3)))
        l14 = self.bn_up4(self.g.nonlinear(self.concat_layer(self.g.upsample_up(self.sn_conv_up4(l13)), conv2)))
        l15 = self.bn_up5(self.g.nonlinear(self.concat_layer(self.g.upsample_up(self.sn_conv_up5(l14)), conv1)))
        l16 = self.bn_up6(self.g.nonlinear(self.sn_conv_up6(l15)))
        out = self.g.conv_up7(l16) + inputx
        out = torch.tanh(out)
        return out
#class Discriminator(nn.Module):
#    def __init__(self):
#        super(Discriminator, self).__init__()
#        self.nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#        self.sn_conv_down1 = SpectralNorm(nn.Conv2d(3, 16, 3, stride=1, padding=1))
#        self.sn_conv_down2 = SpectralNorm(nn.Conv2d(16, 32, 5, stride=2, padding=2))
#        self.sn_conv_down2_ = SpectralNorm(nn.Conv2d(32, 32, 3, stride=2, padding=1))
#        self.sn_conv_down3 = SpectralNorm(nn.Conv2d(32, 64, 5, stride=2, padding=2))
#        #self.sn_conv_down3_ = SpectralNorm(nn.Conv2d(64, 64, 3, stride=2, padding=1))
#        self.sn_conv_down4 = SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, padding=2))
#        self.sn_conv_down5 = SpectralNorm(nn.Conv2d(128, 128, 5, stride=2, padding=2))
#        self.sn_conv_down6 = SpectralNorm(nn.Conv2d(128, 128, 5, stride=2, padding=2))
#        #self.conv_down7 = SpectralNorm(nn.Conv2d(128, 1, 12, stride=1, padding=0))
#        self.conv_down7 = SpectralNorm(nn.Conv2d(128, 1, 4, stride=1, padding=0))
#        #self.conv_down7 = SpectralNorm(nn.Conv2d(128, 1, 8, stride=1, padding=0))
#        #self.sn_conv_down7 = SpectralNorm(nn.Conv2d(128, 1, 16, stride=1, padding=0))
#
#        #self.instance_norm1 = nn.InstanceNorm2d(16)
#        #self.instance_norm2 = nn.InstanceNorm2d(32)
#        #self.instance_norm2_ = nn.InstanceNorm2d(32)
#        #self.instance_norm3 = nn.InstanceNorm2d(64)
#        #self.instance_norm4 = nn.InstanceNorm2d(128)
#        #self.instance_norm5 = nn.InstanceNorm2d(128)
#        #self.instance_norm6 = nn.InstanceNorm2d(128)
#
#        self.instance_norm1 = nn.BatchNorm2d(16)
#        self.instance_norm2 = nn.BatchNorm2d(32)
#        self.instance_norm2_ = nn.BatchNorm2d(32)
#        self.instance_norm3 = nn.BatchNorm2d(64)
#        self.instance_norm4 = nn.BatchNorm2d(128)
#        self.instance_norm5 = nn.BatchNorm2d(128)
#        self.instance_norm6 = nn.BatchNorm2d(128)
#
#    def forward(self, input):
#        out = self.instance_norm1(self.nonlinear(self.sn_conv_down1(input)))
#        out = self.instance_norm2(self.nonlinear(self.sn_conv_down2(out)))
#        out = self.instance_norm2_(self.nonlinear(self.sn_conv_down2_(out)))
#        out = self.instance_norm3(self.nonlinear(self.sn_conv_down3(out)))
#        out = self.instance_norm4(self.nonlinear(self.sn_conv_down4(out)))
#        out = self.instance_norm5(self.nonlinear(self.sn_conv_down5(out)))
#        out = self.instance_norm6(self.nonlinear(self.sn_conv_down6(out)))
#
#        #out = self.instance_norm1(self.sn_conv_down1(input))
#        #out = self.instance_norm2(self.sn_conv_down2(out))
#        #out = self.instance_norm2_(self.sn_conv_down2_(out))
#        #out = self.instance_norm3(self.sn_conv_down3(out))
#        #out = self.instance_norm4(self.sn_conv_down4(out))
#        #out = self.instance_norm5(self.sn_conv_down5(out))
#        #out = self.instance_norm6(self.sn_conv_down6(out))
#        #print(out.shape)
#
#        out = self.conv_down7(out)
#        return out

class Discriminator(nn.Module):
    def __init__(self, if_patch=False,patch=0):
        super(Discriminator, self).__init__()
        self.if_patch=if_patch
        self.patch=patch
        self.nonlinear = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sn_conv_down1 = SpectralNorm(nn.Conv2d(3, 16, 3, stride=1, padding=1))
        self.sn_conv_down2 = SpectralNorm(nn.Conv2d(16, 32, 5, stride=2, padding=2))
        self.sn_conv_down2_ = SpectralNorm(nn.Conv2d(32, 32, 3, stride=2, padding=1))
        self.sn_conv_down3 = SpectralNorm(nn.Conv2d(32, 64, 5, stride=2, padding=2))
        self.sn_conv_down3_ = SpectralNorm(nn.Conv2d(32, 64, 3, stride=2, padding=1))
        self.sn_conv_down4 = SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, padding=2))
        self.sn_conv_down4_ = SpectralNorm(nn.Conv2d(64, 128, 1, stride=1, padding=0))
        self.sn_conv_down5 = SpectralNorm(nn.Conv2d(128, 128, 5, stride=2, padding=2))
        self.sn_conv_down6 = SpectralNorm(nn.Conv2d(128, 128, 5, stride=2, padding=2))
        #self.conv_down7 = SpectralNorm(nn.Conv2d(128, 1, 12, stride=1, padding=0))
        self.conv_down7 = SpectralNorm(nn.Conv2d(128, 1, 4, stride=1, padding=0))
        #self.conv_down7 = SpectralNorm(nn.Conv2d(128, 1, 8, stride=1, padding=0))
        #self.sn_conv_down7 = SpectralNorm(nn.Conv2d(128, 1, 16, stride=1, padding=0))

        #self.instance_norm1 = nn.InstanceNorm2d(16)
        #self.instance_norm2 = nn.InstanceNorm2d(32)
        #self.instance_norm2_ = nn.InstanceNorm2d(32)
        #self.instance_norm3 = nn.InstanceNorm2d(64)
        #self.instance_norm4 = nn.InstanceNorm2d(128)
        #self.instance_norm5 = nn.InstanceNorm2d(128)
        #self.instance_norm6 = nn.InstanceNorm2d(128)

        self.instance_norm1 = nn.BatchNorm2d(16)
        self.instance_norm2 = nn.BatchNorm2d(32)
        self.instance_norm2_ = nn.BatchNorm2d(32)
        self.instance_norm3 = nn.BatchNorm2d(64)
        self.instance_norm3_ = nn.BatchNorm2d(64)
        self.instance_norm4 = nn.BatchNorm2d(128)
        self.instance_norm4_ = nn.BatchNorm2d(128)
        self.instance_norm5 = nn.BatchNorm2d(128)
        self.instance_norm6 = nn.BatchNorm2d(128)

    def forward(self, input):
        out = self.instance_norm1(self.nonlinear(self.sn_conv_down1(input)))
        out = self.instance_norm2(self.nonlinear(self.sn_conv_down2(out)))
        out = self.instance_norm2_(self.nonlinear(self.sn_conv_down2_(out)))
        
        if not self.if_patch:
            #print("======================")
            #print(self.if_patch)
            out = self.instance_norm3(self.nonlinear(self.sn_conv_down3(out)))
            out = self.instance_norm4(self.nonlinear(self.sn_conv_down4(out)))
            out = self.instance_norm5(self.nonlinear(self.sn_conv_down5(out)))
            out = self.instance_norm6(self.nonlinear(self.sn_conv_down6(out)))
            out = self.conv_down7(out)
        elif self.patch==32:
            out = self.instance_norm3_(self.nonlinear(self.sn_conv_down3_(out)))
            out = self.instance_norm4_(self.nonlinear(self.sn_conv_down4_(out)))
            out = self.conv_down7(out)
        elif self.patch==64:
            out = self.instance_norm3(self.nonlinear(self.sn_conv_down3(out)))
            out = self.instance_norm4(self.nonlinear(self.sn_conv_down4(out)))
            out = self.conv_down7(out)
        """
        out = self.instance_norm1(self.sn_conv_down1(input))
        out = self.instance_norm2(self.sn_conv_down2(out))
        out = self.instance_norm2_(self.sn_conv_down2_(out))
        out = self.instance_norm3(self.sn_conv_down3(out))
        out = self.instance_norm4(self.sn_conv_down4(out))
        out = self.instance_norm5(self.sn_conv_down5(out))
        out = self.instance_norm6(self.sn_conv_down6(out))
        """
        #print(out.shape)

        #out = self.conv_down7(out)
        return out
"""
class GneratorX1(nn.Module): #G'x:与Gx共享除BN层外的所有参数
    def __init__(self, Gx):
        pass
    def forward(self, inputx1):
        pass
        
class GneratorY(nn.Module): #Gy
    def __init__(self, Gy):
        pass
    def forward(self, inputy):
        pass
        
class GneratorY1(nn.Module): #G'y:与Gy共享除BN层外的所有参数
    def __init__(self, Gy):
        pass
    def forward(self, inputy1):
        pass
"""





















