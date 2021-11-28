import torch
import torch.nn as nn

class ProWGANDLoss(nn.Module):
    def __init__(self):
        super(ProWGANDLoss, self).__init__()
    
    def gradient_penalty(self, D, real_img, fake_img, step, alpha, LAMDA=10):
        batch_size = real_img.size(0)
        device = real_img.device
        gp_alpha = torch.rand(batch_size, 1)
        gp_alpha = gp_alpha.expand(batch_size, real_img.nelement()//batch_size).reshape(real_img.shape).to(device)
        x = (gp_alpha * real_img + (1 - gp_alpha) * fake_img).requires_grad_(True).to(device)
        out = D(x, step=step, alpha=alpha)

        grad_outputs = torch.ones(out.shape).to(device)

        gradients = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=grad_outputs, create_graph=True, only_inputs=True)[0]
        gradients = gradients.reshape(batch_size, -1)

        return LAMDA * ((gradients.norm(2, dim=1)-1)**2).mean()
    
    def forward(self, G, D, z, img, step, alpha):
        # D loss
        loss_real = -D(img, step=step, alpha=alpha).mean()
        fake_img = G(z, step=step, alpha=alpha)
        loss_fake = D(fake_img.detach(), step=step, alpha=alpha).mean()
        gp = self.gradient_penalty(D, img.detach(), fake_img.detach(), step=step, alpha=alpha, LAMDA=10)
        
        return loss_real + loss_fake + gp
    