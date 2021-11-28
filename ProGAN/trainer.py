import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
import dataset
from model import *
from loss import ProWGANDLoss
from utils import yaml2obj

def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

class Trainer:
    def __init__(self, yaml_filename, max_batch_size=1024):
        assert os.path.exists(yaml_filename), f'Not find this config file: {yaml_filename}'
        print(f'Load config file: {yaml_filename}')
        self.cfg = yaml2obj(yaml_filename)
        
        print(f'Max batch size: {max_batch_size}')
        self.max_batch_size = max_batch_size
        
        print(f'Load dataset: {self.cfg.dataset.name}')
        self.dataset = getattr(dataset, self.cfg.dataset.name)
        
        print('Init G and D models')
        self.G = Generator(z_dim=self.cfg.net.z_dim, in_channels=self.cfg.net.max_channels, max_size=self.cfg.net.max_size).to(self.cfg.train.device)
        self.D = Discriminator(max_size=self.cfg.net.max_size, out_channels=self.cfg.net.max_channels).to(self.cfg.train.device)
        
        # pretrained
        if self.cfg.net.preG_path is not None:
            print('Load pretrained G model')
            self.G = load_dict(self.G, torch.load(self.cfg.net.preG_path).state_dict(), device=self.cfg.train.device)
        if self.cfg.net.preD_path is not None:
            print('Load pretrained D model')
            self.D = load_dict(self.D, torch.load(self.cfg.net.preD_path).state_dict(), device=self.cfg.train.device)
        
        print('Set optimizers and loss function')
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.cfg.train.lr, betas=(0.0, 0.99))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.cfg.train.lr, betas=(0.0, 0.99))
        self.lossfn_D = ProWGANDLoss().to(self.cfg.train.device)
        
        if self.cfg.dataset.fixed_z_path is None:
            self.fixed_z = torch.normal(self.cfg.dataset.noise_mean, self.cfg.dataset.noise_std, size=(64, self.cfg.net.z_dim, 1, 1)).to(self.cfg.train.device)
        else:
            self.fixed_z = torch.load(self.cfg.dataset.fixed_z_path).to(self.cfg.train.device)
        
        # output dirs
        assert self.cfg.output.output_dir is not None, 'The output directory must exist'
        print('Creat output directories')
        self.task_dir = make_dir(os.path.join(self.cfg.output.output_dir, f'{self.cfg.task.name}_{self.cfg.task.id}'))
        self.time_prefix = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        
        self.logs_dir = make_dir(os.path.join(self.task_dir, 'logs'))
        self.imgs_dir = make_dir(os.path.join(self.task_dir, 'imgs', f'{self.time_prefix}_imgs'))
        self.models_dir = make_dir(os.path.join(self.task_dir, 'trained_models', f'{self.time_prefix}_models'))
        
    def get_loader(self, step):
        img_size = 2 ** (step + 2)
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size))
        ])
        img_dataset = self.dataset(data_dir=self.cfg.dataset.data_dir, transforms=transforms_, num=self.cfg.dataset.num_dataset)
        return len(img_dataset), DataLoader(
            img_dataset,
            batch_size=int(np.clip(self.cfg.train.batch_n*1024//4**step, 1, self.max_batch_size)), # n*[1024, 256, 64, 16, 4, 2] when imgsize from 4 to 128
            shuffle=False,
            num_workers=10
        )
        
    def train(self):
        try:
            print('Start training')
            total_steps = int(np.log2(self.cfg.net.max_size) - 1)
            assert self.cfg.train.current_step < total_steps, f'current_step of config must less than {total_steps}'
            for step in range(self.cfg.train.current_step, total_steps):
                num_dataset, loader = self.get_loader(step)
                start_epoch = self.cfg.train.current_epoch if self.cfg.train.current_step == step else 0
                alpha = start_epoch*2/self.cfg.train.epoches if self.cfg.train.current_step == step else 1e-8
                for epoch in range(start_epoch, self.cfg.train.epoches):
                    begin_time = time.time()
                    for i, img in enumerate(tqdm(loader, desc=f'epoch {epoch+1}/{self.cfg.train.epoches}', ncols=100)):
                        z = torch.normal(self.cfg.dataset.noise_mean, self.cfg.dataset.noise_std, size=(img.size(0), self.cfg.net.z_dim, 1, 1))
                        img, z = img.to(self.cfg.train.device), z.to(self.cfg.train.device)

                        # train D
                        self.D.zero_grad()
                        loss_D = self.lossfn_D(self.G, self.D, z, img, step, alpha)
                        loss_D.backward()
                        self.optimizer_D.step()

                        # train G
                        self.G.zero_grad()
                        loss_G = -self.D(self.G(z, step=step, alpha=alpha), step=step, alpha=alpha).mean()
                        loss_G.backward()
                        self.optimizer_G.step()

                        # smooth increase alpha
                        # it reaches 1 after half of epoches
                        alpha += 2 * img.size(0) / (self.cfg.train.epoches * num_dataset)
                        alpha = min(alpha, 1)
                    end_time = time.time()
                    self.log(begin_time, end_time, step, total_steps, epoch, alpha, loss_G, loss_D, log_filename='log.txt')
                    self.test(step, epoch, alpha, save_imgs=True, imshow=False)
                    
                    if (epoch+1) % self.cfg.train.save_every_epoches == 0:
                        self.save_models(step, epoch)
                    
        except Exception as e:
            print('Exception: ', e)
        finally:
            # save models
            self.save_models(step, epoch)
            torch.save(self.fixed_z, os.path.join(self.models_dir, 'fixed_z.pth'))
        
    def test(self, step, epoch, alpha, save_imgs=True, imshow=True):
        with torch.no_grad():
            out_img = make_grid(self.G(self.fixed_z, step=step, alpha=alpha), padding=2, normalize=True).cpu().permute(1,2,0).numpy()
        if imshow:
            fig = plt.figure(figsize=(10,10))
            plt.axis("off")
            plt.imshow(out_img)
            plt.show()
            plt.close()
        if save_imgs:
            out_img = (out_img - np.min(out_img)) * 255 / (np.max(out_img) - np.min(out_img))
            im = Image.fromarray(out_img.astype(np.uint8))
            im.save(os.path.join(self.imgs_dir, 'step{:02d}-epoch{:03d}.png'.format(step+1, epoch+1)))
    
    def log(self, begin_time, end_time, step, total_steps, epoch, alpha, loss_G, loss_D, log_filename=None):
        out_str = '[total time: {:.5f}s] '.format(end_time-begin_time) + f'[Step: {step+1}/{total_steps}] [Epoch: {epoch+1}/{self.cfg.train.epoches}] [alpha: {format(alpha, ".2e")}] [G loss: {loss_G.item()}] [D loss: {loss_D.item()}]'
        print(out_str)
        
        if log_filename is not None:
            log_path = os.path.join(self.logs_dir, f'{self.time_prefix}_{log_filename}')
            with open(log_path, 'a') as f:
                f.write(out_str+'\n')
    
    def save_models(self, step, epoch):
        print('Saving models')
        torch.save(self.G, os.path.join(self.models_dir, f'G_step{step}_epoch{epoch}.pth'))
        torch.save(self.D, os.path.join(self.models_dir, f'D_step{step}_epoch{epoch}.pth'))

if __name__ == '__main__':
    G = torch.load('/data1/cgl/tasks/FFHQ_001/trained_models/20211105_212141_models/G_step4_epoch99.pth')
    D = torch.load('/data1/cgl/tasks/FFHQ_001/trained_models/20211105_212141_models/D_step4_epoch99.pth')
    
    torch.save(G.state_dict(), 'G_dict.pth')
    torch.save(D.state_dict(), 'D_dict.pth')

    
    