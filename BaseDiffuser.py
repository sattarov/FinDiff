import math

import numpy as np
import torch
import torch.nn.functional as F


class BaseDiffuser(object):

    def __init__(
            self, 
            total_steps=1000, 
            beta_start=1e-4, 
            beta_end=0.02, 
            device='cpu',
            scheduler='linear'
        ):
        """Base constructor for diffusion operations

        Args:
            total_steps (int, optional): total diffusion steps. Defaults to 1000.
            beta_start (float), optional): beta start value. Defaults to 1e-4.
            beta_end (float, optional): beta end value. Defaults to 0.02.
            device (str, optional): either cpu or cuda. Defaults to 'cpu'.
            scheduler (str, optional): scheduler type. Defaults to 'linear'.
        """

        self.total_steps = total_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.alphas, self.betas = self.prepare_noise_schedule(scheduler=scheduler)
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def prepare_noise_schedule(self, scheduler: str):
        """ build a noise scheduler based on the provided scheduler type, total steps, and start/end betas

        Args:
            scheduler (str): a scheduler type (linear, quad)

        Raises:
            Exception: wrong scheduler type

        Returns:
            Tensor: corrensponding alphas and betas
        """
        scale = 1000 / self.total_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        if scheduler == 'linear':
            # betas = torch.linspace(self.beta_start, self.beta_end, self.total_steps)
            betas = torch.linspace(beta_start, beta_end, self.total_steps)
            alphas = 1.0 - betas
        elif scheduler == 'quad':
            betas = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.total_steps) ** 2
            alphas = 1.0 - betas
        else:
            raise Exception('Undefined scheduler name')

        return alphas.to(self.device), betas.to(self.device)

    def sample_timesteps(self, n: int):
        """sample list of random timesteps

        Args:
            n (int): number of timesteps to generate

        Returns:
            Tensor: generated list of random timesteps
        """
        t = torch.randint(low=1, high=self.total_steps, size=(n,), device=self.device)
        return t

    def add_gauss_noise(self, x_num, t):
        """ Add gaussian noise to the input data given a specific timestep value

        Args:
            x_num (Tensor): input data tensor
            t (Tensor): list of timesteps

        Returns:
            Tensor: a data tensor with injected noise (x_noise_num) and noise itself (x_noise)
        """
        # numeric attributes
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[t])[:, None]
        noise_num = torch.randn_like(x_num)
        x_noise_num = sqrt_alpha_hat * x_num + sqrt_one_minus_alpha_hat * noise_num
        return x_noise_num, noise_num

    def p_sample_gauss(self, model_out, z_norm, t):
        """ Sampling or denoising step

        Args:
            model_out: trained model used for noise removal
            z_norm (Tensor): initial data tensor
            t (Tensor): list of timesteps

        Returns:
            Tensor: denoised tensor
        """
        sqrt_alpha_t = torch.sqrt(self.alphas[t])[:, None]
        betas_t = self.betas[t][:, None]
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alphas_hat[t])[:, None]
        epsilon_t = torch.sqrt(self.betas[t][:, None])

        random_noise = torch.randn_like(z_norm)
        random_noise[t == 0] = 0.0

        model_mean = ((1 / sqrt_alpha_t) * (z_norm - (betas_t * model_out / sqrt_one_minus_alpha_hat_t)))
        z_norm = model_mean + (epsilon_t * random_noise)

        return z_norm