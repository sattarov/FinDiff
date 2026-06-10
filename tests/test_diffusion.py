import pytest
import torch
from findiff.diffusion import BaseDiffuser

def test_base_diffuser_init():
    diffuser = BaseDiffuser(total_steps=100, beta_start=1e-4, beta_end=0.02, scheduler='linear')
    assert diffuser.total_steps == 100
    assert diffuser.alphas.shape == (100,)
    assert diffuser.betas.shape == (100,)
    assert diffuser.alphas_hat.shape == (100,)

def test_schedulers():
    diffuser_linear = BaseDiffuser(scheduler='linear')
    diffuser_quad = BaseDiffuser(scheduler='quadratic')
    diffuser_sig = BaseDiffuser(scheduler='sigmoid')
    diffuser_exp = BaseDiffuser(scheduler='exponential')

    assert diffuser_linear.betas[0] > 0
    assert diffuser_quad.betas[0] > 0
    assert diffuser_sig.betas[0] > 0
    assert diffuser_exp.betas[0] > 0

def test_sample_timesteps():
    diffuser = BaseDiffuser(total_steps=50)
    t = diffuser.sample_timesteps(10)
    assert t.shape == (10,)
    assert (t >= 0).all() and (t < 50).all()

def test_add_gauss_noise():
    diffuser = BaseDiffuser()
    x = torch.zeros((5, 10))
    t = torch.tensor([0, 10, 50, 100, 999])
    
    x_noise, noise = diffuser.add_gauss_noise(x, t)
    assert x_noise.shape == x.shape
    assert noise.shape == x.shape

def test_p_sample_gauss():
    diffuser = BaseDiffuser()
    model_out = torch.randn((5, 10))
    z_norm = torch.randn((5, 10))
    t = torch.tensor([0, 10, 50, 100, 999])
    
    z_next = diffuser.p_sample_gauss(model_out, z_norm, t)
    assert z_next.shape == z_norm.shape