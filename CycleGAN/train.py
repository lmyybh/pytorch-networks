import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from dataset import ImageDataset
from model import GeneratorResNet, Discriminator
from loss import GeneratorLoss, DiscriminatorLoss

# settings
data_dir = '/home/cgl/data/monet'
batch_size = 10
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epoches = 500 # number of epoches
decay_epoch = 200 # The number of epoch the learning rate starts to decline
n_test_epoches = 50
save_dir = './data/20210818'

# Load data
transforms_ = transforms.Compose([
   # transforms.Resize(int(256*1.12), Image.BICUBIC),
    #transforms.RandomCrop(256, 256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainloader = DataLoader(
    ImageDataset(data_dir, mode='train', transforms=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 10
)

testloader = DataLoader(
    ImageDataset(data_dir, mode='test', transforms=transforms_),
    batch_size = batch_size,
    shuffle = False,
    num_workers = 10
)

# initalize G and D
G_AB = GeneratorResNet(3, num_residual_blocks=9)
D_B = Discriminator(3)

G_BA = GeneratorResNet(3, num_residual_blocks=9)
D_A = Discriminator(3)

# define loss
G_loss = GeneratorLoss(thetas=[5.0, 1.0, 10.0])
D_loss = DiscriminatorLoss()

# cuda
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
if cuda:
    device = 'cuda:0'
    G_AB = G_AB.to(device)
    D_B = D_B.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    
    G_loss = G_loss.to(device)
    D_loss = D_loss.to(device)

# optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

# schedulers
lambda_func = lambda epoch: 1 - max(0, epoch-decay_epoch)/(n_epoches-decay_epoch)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_func)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_func)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_func)

# train    
for epoch in range(n_epoches):
    for i, (real_A, real_B) in enumerate(trainloader):
        real_A, real_B = real_A.type(Tensor), real_B.type(Tensor)
        
        # groud truth
        out_shape = [real_A.size(0), 1, real_A.size(2)//D_A.scale_factor, real_A.size(3)//D_A.scale_factor]
        valid = torch.ones(out_shape).type(Tensor)
        fake = torch.zeros(out_shape).type(Tensor)
        
        """Train Generators"""
        # set to training mode in the begining, beacause sample_images will set it to eval mode
        G_AB.train()
        G_BA.train()
        
        optimizer_G.zero_grad()
        
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)
        D_fake_B = D_B(fake_B)
        D_fake_A = D_A(fake_A)
        loss_G = G_loss(real_A, real_B, fake_A, fake_B, recov_A, recov_B, D_fake_A, D_fake_B)
        
        loss_G.backward()
        optimizer_G.step()
        
        """Train Discriminator A"""
        optimizer_D_A.zero_grad()
        
        loss_D_A = D_loss(D_A(real_A), D_A(fake_A.detach()))
        
        loss_D_A.backward()
        optimizer_D_A.step()
        
        """Train Discriminator B"""
        optimizer_D_B.zero_grad()
        
        loss_D_B = D_loss(D_B(real_B), D_B(fake_B.detach()))
        
        loss_D_B.backward()
        optimizer_D_B.step()
    
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    # test
    loss_D = (loss_D_A + loss_D_B) / 2
    out_str = f'[Epoch {epoch+1}/{n_epoches}] [G loss: {loss_G.item()} | identity: {G_loss.losses[0].item()} GAN: {G_loss.losses[1].item()} cycle: {G_loss.losses[2].item()}] [D loss: {loss_D.item()} | D_A: {loss_D_A.item()} D_B: {loss_D_B.item()}]'
    print(out_str)
    
    if (epoch+1) % n_test_epoches == 0:
        print('='*50 + ' save models and log ' + '='*50)
        # save models
        models_dir = os.path.join(save_dir, 'pth', f'epoch-{epoch+1}')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        torch.save(G_AB, os.path.join(models_dir, 'G_AB.pth'))
        torch.save(G_BA, os.path.join(models_dir, 'G_BA.pth'))
        torch.save(D_A, os.path.join(models_dir, 'D_A.pth'))
        torch.save(D_B, os.path.join(models_dir, 'D_B.pth'))
        
        # save logs
        log_file_path = os.path.join(save_dir, 'log.txt')
        with open(log_file_path, 'a') as f:
            f.write(out_str + '\n')
        
        