import torch
import torch.nn as nn

class GeneratorLoss(nn.Module):
    def __init__(self, thetas=[5.0, 1.0, 10.0]):
        super(GeneratorLoss, self).__init__()
        self.thetas = thetas
        
    def forward(self, real_A, real_B, fake_A, fake_B, recov_A, recov_B, D_fake_A, D_fake_B):
        criterion_GAN = nn.MSELoss()
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()
        
        # identity loss
        loss_id_A = criterion_identity(fake_B, real_A)
        loss_id_B = criterion_identity(fake_A, real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        
        # GAN loss, train G to make D think it's true
        valid = torch.ones_like(D_fake_A)
        loss_GAN_AB = criterion_GAN(D_fake_B, valid)
        loss_GAN_BA = criterion_GAN(D_fake_A, valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        
        # cycle loss
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
        self.losses = [loss_identity, loss_GAN, loss_cycle]
        
        return self.thetas[0]*loss_identity + self.thetas[1]*loss_GAN + self.thetas[2]*loss_cycle
     

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
    
    def forward(self, real_pred, fake_pred):
        # gold truth
        valid = torch.ones_like(real_pred)
        fake = torch.zeros_like(fake_pred)
        
        criterion_GAN = nn.MSELoss()
        loss_real = criterion_GAN(real_pred, valid)
        loss_fake = criterion_GAN(fake_pred, fake)
        
        return (loss_real + loss_fake) / 2

        
        
        
        