import os
import pytest
import pandas as pd
import numpy as np
from findiff.data import DataTransformer, FinDiffDataset

def test_data_transformer_fit_transform():
    df = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0, 4.0],
        'cat1': ['a', 'b', 'a', 'c']
    })
    y = pd.Series([0, 1, 0, 1])

    dt = DataTransformer(
        numerical_cols=['num1'],
        categorical_cols=['cat1']
    )
    res = dt.fit_transform(df, y)
    
    assert 'num' in res
    assert 'cat' in res
    assert 'label' in res
    assert res['num'].shape == (4, 1)
    assert res['cat'].shape == (4, 1)
    assert res['label'].shape == (4,)

def test_data_transformer_inverse_transform():
    df = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0],
        'cat1': ['a', 'b', 'c']
    })
    dt = DataTransformer(numerical_cols=['num1'], categorical_cols=['cat1'])
    res = dt.fit_transform(df)
    inv_df = dt.inverse_transform(res)
    
    assert list(inv_df.columns) == ['cat1', 'num1']
    assert inv_df['cat1'].tolist() == ['a', 'b', 'c']
    np.testing.assert_allclose(inv_df['num1'].to_numpy(dtype=float), [1.0, 2.0, 3.0])

def test_save_load_transformer(tmp_path):
    df = pd.DataFrame({'num1': [1.0, 2.0], 'cat1': ['a', 'b']})
    dt = DataTransformer(numerical_cols=['num1'], categorical_cols=['cat1'])
    dt.fit(df)
    
    path = os.path.join(tmp_path, "dt.pkl")
    dt.save(path)
    
    dt_loaded = DataTransformer.load(path)
    assert dt_loaded.fitted is True
    assert dt_loaded.numerical_cols == ['num1']
    assert dt_loaded.categorical_cols == ['cat1']

def test_findiff_dataset():
    cat_data = np.array([[0], [1]])
    num_data = np.array([[0.5], [1.5]])
    labels = np.array([0, 1])
    
    dataset = FinDiffDataset(cat_data, num_data, labels)
    assert len(dataset) == 2
    
    sample = dataset[0]
    np.testing.assert_array_equal(sample["cat"], [0])
    np.testing.assert_array_equal(sample["num"], [0.5])
    assert sample["label"] == 0