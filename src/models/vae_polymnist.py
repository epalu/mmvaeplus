# PolyMNIST model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from utils import Constants
from .vae import VAE
from datasets_PolyMNIST import PolyMNISTDataset

# Constants
dataSize = torch.Size([3, 28, 28])

def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out

class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Encoder network
class Enc(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, ndim_w, ndim_z):
        super().__init__()
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks_w = [
            ResnetBlock(nf, nf)
        ]

        blocks_z = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img_w = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet_w = nn.Sequential(*blocks_w)
        self.fc_mu_w = nn.Linear(self.nf0*s0*s0, ndim_w)
        self.fc_lv_w = nn.Linear(self.nf0*s0*s0, ndim_w)

        self.conv_img_z = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_z = nn.Sequential(*blocks_z)
        self.fc_mu_z = nn.Linear(self.nf0 * s0 * s0, ndim_z)
        self.fc_lv_z = nn.Linear(self.nf0 * s0 * s0, ndim_z)

    def forward(self, x):
        out_w = self.conv_img_w(x)
        out_w = self.resnet_w(out_w)
        out_w = out_w.view(out_w.size()[0], self.nf0*self.s0*self.s0)
        lv_w = self.fc_lv_w(out_w)

        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.view(out_z.size()[0], self.nf0 * self.s0 * self.s0)
        lv_z = self.fc_lv_z(out_z)

        return torch.cat((self.fc_mu_w(out_w), self.fc_mu_z(out_z)), dim=-1), \
               torch.cat((F.softmax(lv_w, dim=-1) * lv_w.size(-1) + Constants.eta,
                          F.softmax(lv_z, dim=-1) * lv_z.size(-1) + Constants.eta), dim=-1)

# Decoder network
class Dec(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, ndim):
        super().__init__()

        # NOTE: I've set below variables according to Kieran's suggestions
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(ndim, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, u):
        out = self.fc(u).view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        out = out.view(*u.size()[:2], *out.size()[1:])
        # consider also predicting the length scale
        return out, torch.tensor(0.75).to(u.device)  # mean, length scale


class PolyMNIST(VAE):
    """ Derive a specific sub-class of a VAE for SVHN """

    def __init__(self, params):
        super(PolyMNIST, self).__init__(
            dist.Laplace,  # prior
            dist.Laplace,  # likelihood
            dist.Laplace,  # posterior
            Enc(params.latent_dim_w, params.latent_dim_z),
            Dec(params.latent_dim_w + params.latent_dim_z),
            params
        )
        self._pu_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False)  # logvar
        ])
        grad_w = {'requires_grad': True}
        self._pw_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), **grad_w)  # logvar
        ])
        self.modelName = 'polymnist_resnet'
        self.dataSize = dataSize
        self.llik_scaling = 1.
        self.params = params

    @property
    def pu_params(self):
        return self._pu_params[0], F.softmax(self._pu_params[1], dim=1) * self._pu_params[1].size(-1)

    @property
    def pw_params(self):
        return self._pw_params[0], F.softmax(self._pw_params[1], dim=1) * self._pw_params[1].size(-1)
    '''
    @property
    def pu_params(self):
        return self._pu_params[0], F.softplus(self._pu_params[1]) + Constants.eta

    @property
    def pw_params(self):
        return self._pw_params[0], F.softplus(self._pw_params[1]) + Constants.eta
    '''

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', m=0):
        unim_train_datapaths = [self.tmpdir + "/PolyMNIST/train/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
        unim_test_datapaths = [self.tmpdir + "/PolyMNIST/test/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        tx = transforms.ToTensor()
        train = DataLoader(PolyMNISTDataset(unim_train_datapaths, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(PolyMNISTDataset(unim_test_datapaths, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def generate(self, runPath, epoch): # NOT EXPOSED: we only train multimodal VAEs here
        N, K = 64, 9
        samples = super(PolyMNIST, self).generate(N, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):  # NOT EXPOSED: we only train multimodal VAEs here
        recon = super(PolyMNIST, self).reconstruct(data)
        comp = torch.cat([data, recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(runPath, epoch))

