import torch
import torch.nn as nn
import torch.optim as optimizer
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_channel, feat_d):
        super(Discriminator, self).__init__()
        # input_size = 3*128*128
        self.disc = nn.Sequential(
            nn.Conv2d(img_channel, feat_d, kernel_size=4, stride=2, padding=1, bias=True),  # feat_d*64*64
            nn.LeakyReLU(0.2),
            self._block(feat_d, 2*feat_d, 4, 2, 1),  # 2*feat_d*32*32
            self._block(2*feat_d, 4*feat_d, 4, 2, 1), # 4*feat*16*16
            self._block(4*feat_d, 8*feat_d, 4, 2, 1), # 8*feat*8*8
            self._block(8*feat_d, 16*feat_d, 4, 2, 1), # 16*feat_d*4*4
            nn.Conv2d(in_channels=16*feat_d, out_channels=1, kernel_size=4, stride=2, padding=0), # 1*1*1
            nn.Sigmoid()
        )
                
    def _block(self, inp_cha, out_chan, kern_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=inp_cha,
                out_channels=out_chan,
                kernel_size=kern_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, img_channel, z_dim, feat_g):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            self._block(z_dim, feat_g*32, 4, 1, 0), # img: 4*4
            self._block(feat_g*32, feat_g*16, 4, 2, 1), # img: 8*8
            self._block(feat_g*16, feat_g*8, 4, 2, 1), #img: 16*16
            self._block(feat_g*8, feat_g*4, 4, 2, 1), # img: 32*32
            self._block(feat_g*4, feat_g*2, 4, 2, 1), # img: 64*64
            nn.ConvTranspose2d(in_channels=feat_g*2, out_channels=img_channel, kernel_size=4, stride=2, padding=1), # img 128*128
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kern_size, stride, padd):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kern_size, 
                stride=stride,
                padding=padd, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.gen(x)