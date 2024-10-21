import torch
import torch.nn as nn


def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    #cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
    cr = nn.CrossEntropyLoss()
    return cr(output, target)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        # cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
        cr = nn.CrossEntropyLoss(reduction='none')
        return torch.mean(cr(output, target))


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        # cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
        mse = nn.MSELoss()
        return torch.mean(mse(output, target))
class MSE_KLD_Loss(nn.Module):
    def __init__(self):
        super(MSE_KLD_Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, recon_x, x, mu, logvar):
        MSE = torch.mean(self.mse_loss(recon_x, x))

        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        print("KLD.shape",KLD.shape)
        a = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        print("a",x.shape)
        print("a", mu.shape)
        print("a",logvar.shape)

        return MSE +  KLD #0.25 *

