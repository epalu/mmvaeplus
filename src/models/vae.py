# Base VAE class definition

import torch
import torch.nn as nn

from utils import get_mean


class VAE(nn.Module):
    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pu = prior_dist
        self.px_u = likelihood_dist
        self.qu_x = post_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self._pu_params = None  # defined in subclass
        self._qu_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0

        self._pw_params = None # defined in subclass

    @property
    def pu_params(self):
        return self._pu_params

    @property
    def pw_params(self):
        return self._pw_params

    @property
    def qu_x_params(self):
        if self._qu_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qu_x_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        self._qu_x_params = self.enc(x)
        qu_x = self.qu_x(*self._qu_x_params)
        us = qu_x.rsample(torch.Size([K]))
        px_u = self.px_u(*self.dec(us))
        return qu_x, px_u, us

    def generate(self, N, K):   # Not exposed as here we only train multimodal VAES
        self.eval()
        with torch.no_grad():
            pu = self.pu(*self.pu_params)
            latents = pu.rsample(torch.Size([N]))
            px_u = self.px_u(*self.dec(latents))
            data = px_u.sample(torch.Size([K]))
        return data.view(-1, *data.size()[3:])

    def reconstruct(self, data):  # Not exposed as here we only train multimodal VAES
        self.eval()
        with torch.no_grad():
            qu_x = self.qu_x(*self.enc(data))
            latents = qu_x.rsample()  # no dim expansion
            px_u = self.px_u(*self.dec(latents))
            recon = get_mean(px_u)
        return recon