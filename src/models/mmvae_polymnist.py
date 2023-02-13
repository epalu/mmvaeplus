# PolyMNIST-PolyMNIST multi-modal model specification
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from datasets_PolyMNIST import PolyMNISTDataset
from .mmvae import MMVAE
from .vae_polymnist import PolyMNIST


class PolyMNIST_5modalities(MMVAE):
    def __init__(self, params):
        super(PolyMNIST_5modalities, self).__init__(dist.Laplace, params, PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST)
        self._pu_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False)  # logvar
        ])
        # REMOVE LLIK SCALING
        # self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            # if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'polymnist-5modalities'
        # Fix model names for indiviudal models to be saved
        for idx, vae in enumerate(self.vaes):
            vae.modelName = 'polymnist_m'+str(idx)
            vae.llik_scaling = 1.0
        self.tmpdir = params.tmpdir

    @property
    def pu_params(self):
        return self._pu_params[0], F.softmax(self._pu_params[1], dim=1) * self._pu_params[1].size(-1)

    #@property
    #def pu_params(self):
    #    return self._pu_params[0], F.softplus(self._pu_params[1]) + Constants.eta

    #def setTmpDir(self, tmpdir):
    #    self.tmpdir = tmpdir

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        tx = transforms.ToTensor()
        unim_train_datapaths = [self.tmpdir+"/PolyMNIST/train/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
        unim_test_datapaths = [self.tmpdir+"/PolyMNIST/test/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
        dataset_PolyMNIST_train = PolyMNISTDataset(unim_train_datapaths, transform=tx)
        dataset_PolyMNIST_test = PolyMNISTDataset(unim_test_datapaths, transform=tx)
        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(dataset_PolyMNIST_train, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(dataset_PolyMNIST_test, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def generate_tb(self):
        N = 100
        outputs = []
        samples_list = super(PolyMNIST_5modalities, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            samples = samples.view(N, *samples.size()[1:])
            outputs.append(make_grid(samples, nrow=int(sqrt(N))))
        return outputs

    def generate_for_calculating_unconditional_coherence(self, N):
        samples_list = super(PolyMNIST_5modalities, self).generate(N)
        return [samples.data.cpu() for samples in samples_list]

    def generate_for_fid(self, savedir, num_samples, tranche):
        N = num_samples
        samples_list = super(PolyMNIST_5modalities, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            for image in range(samples.size(0)):
                save_image(samples[image, :, :, :], '{}/random/m{}/{}_{}.png'.format(savedir, i, tranche, image))

    def reconstruct_for_fid_tb(self, data, savedir, i):
        recons_mat = super(PolyMNIST_5modalities, self).reconstruct_and_cross_reconstruct([d for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                recon = recon.squeeze(0).cpu()
                for image in range(recon.size(0)):
                    save_image(recon[image, :, :, :],
                                '{}/m{}/m{}/{}_{}.png'.format(savedir, r,o, image, i))

    def cross_generate_tb(self, data):
        N = 10
        recon_triess = [[[] for i in range(N)] for j in range(N)]
        outputss = [[[] for i in range(N)] for j in range(N)]
        for i in range(10):
            recons_mat = super(PolyMNIST_5modalities, self).reconstruct_and_cross_reconstruct([d[:N] for d in data])
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                      recon = recon.squeeze(0).cpu()
                      recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(torch.cat([data[r][:N].cpu()]+recon_triess[r][o]), nrow=N)
        return outputss

def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
