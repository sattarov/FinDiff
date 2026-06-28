import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from findiff.model import FinDiff


@pytest.fixture
def mock_dt():
    mock_data_transformer = MagicMock()
    mock_data_transformer.categorical_cols = ['col1', 'col2']
    mock_data_transformer.numerical_cols = ['num1']
    mock_data_transformer.categorical_mapping_ = {'col1': {1, 2}, 'col2': {1, 2, 3}}
    mock_data_transformer.label_cardinality_ = None
    mock_data_transformer.embedding_mapping_ = {
        'col1': np.array([0, 1]),
        'col2': np.array([2, 3, 4])
    }
    return mock_data_transformer


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.nn.MSELoss')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_decode_cat_emb_logits(mock_cos, mock_adam, mock_mse, mock_diffuser, mock_synth, mock_dt):
    # 2. Instantiate FinDiff
    model = FinDiff(data_transformer=mock_dt)
    
    # 3. Create input test data
    # Two categorical columns with batch size of 2
    cat_logits = [
        torch.tensor([[10.0, 1.0], [1.0, 10.0]]),            # Argmax: 0, 1
        torch.tensor([[1.0, 10.0, 1.0], [1.0, 1.0, 10.0]])   # Argmax: 1, 2
    ]
    
    # 4. Run the method and assert
    result = model.decode_cat_emb_logits(cat_logits)
    expected = np.array([[0, 3], [1, 4]])
    np.testing.assert_array_equal(result, expected)


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_findiff_init(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, cat_emb_dim=3)
    
    assert model.cat_cols_dim == 2
    assert model.cat_emb_dim == 3
    mock_synth.assert_called_once()
    mock_diffuser.assert_called_once()


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_fit(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, num_epochs=2)
    model.train_epoch = MagicMock(return_value=0.25)
    
    dummy_dataloader = [1, 2]
    model.fit(dummy_dataloader)
    
    assert model.train_epoch.call_count == 2
    assert model.training_history["train_loss"] == [0.25, 0.25]


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_train_epoch(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, cat_decoding='distance')
    
    model.synthesizer = MagicMock()
    model.synthesizer.embed_x_cat.return_value = torch.zeros((2, 4))
    model.synthesizer.return_value = torch.zeros((2, 5))
    
    model.diffuser = MagicMock()
    model.diffuser.sample_timesteps.return_value = torch.tensor([1, 2])
    model.diffuser.add_gauss_noise.return_value = (torch.zeros((2, 5)), torch.zeros((2, 5)))
    
    mock_loss = MagicMock()
    mock_loss.detach.return_value.cpu.return_value.numpy.return_value = 0.5
    model.loss_fnc = MagicMock()
    model.loss_fnc.return_value.sum.return_value.mean.return_value = mock_loss
    
    optimizer = MagicMock()
    scheduler = MagicMock()
    
    batch = {
        "cat": torch.tensor([[0, 2], [1, 3]]),
        "num": torch.tensor([[0.1], [0.2]])
    }
    dataloader = [batch]
    
    loss = model.train_epoch(dataloader, optimizer, scheduler)
    
    optimizer.zero_grad.assert_called_once()
    optimizer.step.assert_called_once()
    scheduler.step.assert_called_once()
    assert loss == 0.5


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
@patch.object(FinDiff, 'decode_sample')
def test_sample(mock_decode, mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, batch_size_sample=2)
    model.synthesizer = MagicMock()
    model.synthesizer.dim_input = 5
    model.synthesizer.return_value = torch.zeros((2, 5))
    
    model.diffuser = MagicMock()
    model.diffuser.total_steps = 3
    model.diffuser.p_sample_gauss.return_value = torch.zeros((2, 5))
    
    mock_decode.return_value = pd.DataFrame({'col1': [1, 2], 'num1': [0.1, 0.2]})
    
    df = model.sample(n_samples=2)
    
    assert len(df) == 2
    mock_decode.assert_called_once()
    assert model.diffuser.p_sample_gauss.call_count == 3


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_decode_sample(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, cat_emb_dim=2)
    model.decode_cat_emb_distance = MagicMock(return_value=np.array([[0, 2]]))
    
    mock_dt.inverse_transform.return_value = pd.DataFrame({'col1': [1], 'num1': [0.1]})
    
    sample = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    res = model.decode_sample(sample)
    
    model.decode_cat_emb_distance.assert_called_once()
    mock_dt.inverse_transform.assert_called_once()
    assert isinstance(res, pd.DataFrame)


@patch('findiff.model.FinDiffSynthesizer')
@patch('findiff.model.BaseDiffuser')
@patch('findiff.model.optim.Adam')
@patch('findiff.model.CosineAnnealingLR')
def test_decode_cat_emb_distance(mock_cos, mock_adam, mock_diffuser, mock_synth, mock_dt):
    model = FinDiff(data_transformer=mock_dt, cat_emb_dim=2, batch_size=2)
    model.synthesizer = MagicMock()
    
    emb_weights = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [10.0, 10.0],
        [11.0, 11.0],
        [12.0, 12.0],
    ])
    model.synthesizer.get_x_cat_emb.return_value = emb_weights
    
    sample_cat = torch.tensor([
        [0.1, 0.1, 10.2, 10.2],
        [0.9, 0.9, 11.8, 11.8],
    ])
    
    res = model.decode_cat_emb_distance(sample_cat)
    
    expected = np.array([
        [0, 2],
        [1, 4]
    ])
    np.testing.assert_array_equal(res, expected)