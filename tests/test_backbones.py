import pytest
import torch
from findiff.backbones import (
    SinusoidalPositionEmbeddings,
    MLPBackbone,
    TransformerBackbone,
    UNetBackbone,
    FinDiffSynthesizer
)

def test_sinusoidal_embeddings():
    embedder = SinusoidalPositionEmbeddings(dim=64)
    time = torch.tensor([0, 10, 100])
    emb = embedder(time)
    assert emb.shape == (3, 64)

def test_mlp_backbone():
    backbone = MLPBackbone(num_features=10, cond_dim=16, hidden_dim=32, num_blocks=2)
    x = torch.randn(4, 10)
    cond = torch.randn(4, 16)
    out = backbone(x, cond)
    assert out.shape == (4, 10)

def test_transformer_backbone():
    backbone = TransformerBackbone(num_features=10, cond_dim=16, d_model=32, n_head=2, num_layers=2, dim_feedforward=64)
    x = torch.randn(4, 10)
    cond = torch.randn(4, 16)
    out = backbone(x, cond)
    assert out.shape == (4, 10)

def test_unet_backbone():
    backbone = UNetBackbone(num_features=10, cond_dim=16, hidden_dims=[32, 64])
    x = torch.randn(4, 10)
    cond = torch.randn(4, 16)
    out = backbone(x, cond)
    assert out.shape == (4, 10)

def test_findiff_synthesizer():
    vocab = {'cat1': {0, 1}, 'cat2': {2, 3, 4}}
    model = FinDiffSynthesizer(
        num_cols_dim=5,
        cat_vocab=vocab,
        cat_emb_dim=4,
        time_embed_dim=16,
        backbone_type='mlp'
    )
    
    # x dimension should be num_cols_dim + len(cat_vocab) * cat_emb_dim = 5 + 2 * 4 = 13
    x = torch.randn(2, 13)
    t = torch.tensor([1, 10])
    
    out = model(x, t)
    assert out.shape == (2, 13)

def test_findiff_synthesizer_logits():
    vocab = {'cat1': {0, 1}, 'cat2': {2, 3, 4}}
    model = FinDiffSynthesizer(
        num_cols_dim=5,
        cat_vocab=vocab,
        cat_emb_dim=4,
        time_embed_dim=16,
        backbone_type='mlp',
        cat_decoding='logits'
    )
    
    x = torch.randn(2, 13)
    t = torch.tensor([1, 10])
    
    out_x, cat_logits = model(x, t)
    assert out_x.shape == (2, 13)
    assert len(cat_logits) == 2
    assert cat_logits[0].shape == (2, 2)
    assert cat_logits[1].shape == (2, 3)

def test_embed_x_cat():
    vocab = {'cat1': {0, 1}, 'cat2': {2, 3, 4}}
    model = FinDiffSynthesizer(num_cols_dim=5, cat_vocab=vocab, cat_emb_dim=4)
    
    x_cat = torch.tensor([[0, 2], [1, 4]])
    emb = model.embed_x_cat(x_cat)
    assert emb.shape == (2, 8)