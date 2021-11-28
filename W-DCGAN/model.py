import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, signal_size, out_channels=3):
        super(Generator, self).__init__()
        self.linear = nn.Linear(signal_size, 1024*4*4)
        
        convs = []
        channels = [1024, 512, 256, 128]
        for i in range(1, len(channels)):
            convs.append(nn.ConvTranspose2d(channels[i-1], channels[i], 2, stride=2))
            convs.append(nn.BatchNorm2d(channels[i]))
            convs.append(nn.LeakyReLU(0.2, inplace=True))
        convs.append(nn.ConvTranspose2d(channels[-1], out_channels, 2, stride=2))
        convs.append(nn.Tanh())
        
        self.convs = nn.Sequential(*convs)
        
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 1024, 4, 4)
        x = self.convs(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        channels = [3, 32, 64, 128, 256, 512, 1024]
        convs = []
        for i in range(1, len(channels)):
            convs.append(nn.Conv2d(channels[i-1], channels[i], 3, padding=1, stride=2))
            convs.append(nn.BatchNorm2d(channels[i]))
            convs.append(nn.LeakyReLU(0.2, inplace=True))
        self.convs = nn.Sequential(*convs)
        self.linear = nn.Linear(1024*1*1, 1)
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)      
        x = self.linear(x)
        # x = torch.sigmoid(x)
        return x

    
