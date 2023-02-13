# Base MMVAE class definition
import torch
import torch.nn as nn
from utils import get_mean


class MMVAE(nn.Module):
    def __init__(self, prior_dist, params, *vaes):
        super(MMVAE, self).__init__()
        self.pu = prior_dist
        self.pw = prior_dist
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])
        self.modelName = None  # filled-in per sub-class
        self.params = params
        self._pu_params = None  # defined in subclass

    @property
    def pu_params(self):
        return self._pu_params

    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(x[m], K=K)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    if self.params.variant == 'mmvaeplus':
                        _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                        pw = self.pw(*vae.pw_params)
                        latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                        if not self.params.no_cuda and torch.cuda.is_available():
                            latents_w.cuda()
                        us_combined = torch.cat((latents_w, z_e), dim=-1)
                        px_us[e][d] = vae.px_u(*vae.dec(us_combined))
                    elif self.params.variant == 'mmvaefactorized':
                        _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                        us_target = uss[d]
                        w_d, _ = torch.split(us_target, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                        us_combined = torch.cat((w_d, z_e), dim=-1)
                        px_us[e][d] = vae.px_u(*vae.dec(us_combined))
                    else:
                        raise ValueError("wrong option for variant paramter")
        return qu_xs, px_us, uss

    def generate(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            pu = self.pu(*self.pu_params)
            latents = pu.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_u = vae.px_u(*vae.dec(latents))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def reconstruct_and_cross_reconstruct_forw(self, data):
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # pw = self.pz(torch.zeros(1, self.params.latent_dim_w), torch.ones(1, self.params.latent_dim_w))
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(data[m], K=1)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        for e, us in enumerate(uss):
            latents_w, latents_z = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
            pu = self.pu(*self.pu_params)
            latents_u_to_split = pu.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
            latents_w_new, _ = torch.split(latents_u_to_split, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
            us = torch.cat((latents_w_new, latents_z), dim=-1)
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_us[e][d] = vae.px_u(*vae.dec(us))
        return qu_xs, px_us, uss

    def reconstruct_and_cross_reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_us, _ = self.reconstruct_and_cross_reconstruct_forw(data)
            # ------------------------------------------------
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_u) for px_u in r] for r in px_us]
        return recons