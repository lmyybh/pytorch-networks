import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WSConv2d(nn.Module):
    """
    Weight scaled Conv2d (Equalized Learning Rate)
    Note that input is multiplied rather than changing weights
    this will have the same result.
    Inspired and looked at:
    https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
    
class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = eps
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
    
# (Conv3*3 + LReLU + PixelNorm) * 2
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            WSConv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
            WSConv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
        )
    
    def forward(self, x):
        return self.block(x)

# ToRGB
class ToRGB(nn.Module):
    def __init__(self, in_channels):
        super(ToRGB, self).__init__()
        self.rgb = WSConv2d(in_channels, 3, 1)
        
    def forward(self, x):
        return self.rgb(x)

class Generator(nn.Module):
    def __init__(self, z_dim=512, in_channels=512, max_size=128):
        super(Generator, self).__init__()
        self.input = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
            WSConv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
        ) # out: in_channels * 4 * 4
        
        steps = int(np.log2(max_size) - 2)
        self.blocks = nn.ModuleList([])
        self.rgbs = nn.ModuleList([ToRGB(in_channels)])
        channels = in_channels
        for i in range(steps):
            if i < 3:
                self.blocks.append(ConvBlock(channels, channels))
                self.rgbs.append(ToRGB(channels))
            else:
                self.blocks.append(ConvBlock(channels, channels//2))
                self.rgbs.append(ToRGB(channels//2))
                channels = channels // 2
                
    def fade_in(self, block, up, alpha):
        return torch.tanh(alpha * block + (1 - alpha) * up)
    
    def forward(self, z, step, alpha):
        x = self.input(z)
        if step == 0:
            return self.rgbs[step](x)
        
        for i in range(step-1):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.blocks[i](x)
        
        x1 = F.interpolate(x, scale_factor=2, mode='nearest')
        block = self.rgbs[step](self.blocks[step-1](x1))
        
        up = self.rgbs[step-1](x)
        up = F.interpolate(up, scale_factor=2, mode='nearest')
        
        return self.fade_in(block, up, alpha)

def minibatch_standard_deviation(x):
    mean_std = torch.std(x, dim=0).mean()
    mean_std = mean_std.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
    return torch.cat([x, mean_std], dim=1)
    
    
class FromRGB(nn.Module):
    def __init__(self, out_channels):
        super(FromRGB, self).__init__()
        self.rgb = WSConv2d(3, out_channels, 1)
        
    def forward(self, x):
        return self.rgb(x)

class Discriminator(nn.Module):
    def __init__(self, max_size=128, out_channels=512):
        super(Discriminator, self).__init__()
        
        self.out = nn.Sequential(
            WSConv2d(out_channels+1, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            WSConv2d(out_channels, out_channels, 4)
        )
        self.linear = nn.Linear(out_channels, 1)
        
        num_blocks = int(np.log2(max_size) - 2)
        self.blocks = nn.ModuleList([])
        self.from_rgbs = nn.ModuleList([FromRGB(out_channels)])
        channels = out_channels
        for i in range(num_blocks):
            if i < 3:
                self.from_rgbs.append(FromRGB(channels))
                self.blocks.append(ConvBlock(channels, channels))
            else:
                self.from_rgbs.append(FromRGB(channels//2))
                self.blocks.append(ConvBlock(channels//2, channels))
                channels = channels // 2
        
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def fade_in(self, block, down, alpha):
        return torch.tanh(alpha * block + (1 - alpha) * down)
    
    def forward(self, x, step, alpha):
        if step == 0:
            x = self.from_rgbs[0](x)
            x = minibatch_standard_deviation(x)
            x = self.out(x).view(x.shape[0], -1)
            return self.linear(x)
        
        block = self.down(self.blocks[step-1](self.from_rgbs[step](x)))
        down = self.from_rgbs[step-1](self.down(x))
        x = self.fade_in(block, down, alpha)
        
        for i in range(step-2, -1, -1):
            x = self.down(self.blocks[i](x))
            
        x = minibatch_standard_deviation(x)
        x = self.out(x).view(x.shape[0], -1)
        return self.linear(x)

def load_dict(net, model_dict, device=None):
    net_dict = net.state_dict()
    state_dict = {k:v for k,v in model_dict.items() if k in net_dict.keys()}
    net_dict.update(state_dict)
    net.load_state_dict(net_dict)
    if device is not None:
        net = net.to(device)
    return net
