# FinDiff

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sattarov/FinDiff/blob/master/main.ipynb)

Implementation of the paper: **"FinDiff: Diffusion Models for Financial Tabular Data Generation"** [link](https://dl.acm.org/doi/abs/10.1145/3604237.3626876).

FinDiff is a PyTorch-based framework for generating synthetic tabular data, specifically designed to handle the complexities of financial datasets (e.g., mixtures of continuous and categorical variables). It leverages diffusion models with swappable neural network backbones to synthesize realistic records.

## Features

- **Multiple Neural Backbones**: Choose between MLP, Transformer, and U-Net backbones for the diffusion process.
- **Robust Data Transformation**: Built-in `DataTransformer` handles standardizing numerical features (Standard, MinMax, Robust, Power, Quantile) and encoding categorical features.
- **Flexible Categorical Handling**: Learnable embeddings for categorical data, with support for decoding via distance-based matching or direct logit prediction.
- **Customizable Diffusion Schedulers**: Configurable noise schedules including linear, quadratic, sigmoid, and exponential.
- **Conditional Generation**: Support for label-conditioned synthetic data generation using Classifier-Free Guidance.

## Usage Example

Below is a minimal example demonstrating how to train a FinDiff model and generate synthetic tabular data.

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader
from findiff.data import DataTransformer, FinDiffDataset
from findiff.model import FinDiff

# 1. Prepare and transform your data
df = pd.DataFrame({
    "cat_col": ["A", "B", "A", "C"],
    "num_col": [1.5, 2.3, 0.9, 3.1]
})

transformer = DataTransformer(
    categorical_cols=["cat_col"], 
    numerical_cols=["num_col"],
    numerical_scaler="standard"
)
transformed_data = transformer.fit_transform(df)

# 2. Create a DataLoader
dataset = FinDiffDataset(
    cat_dataset=torch.tensor(transformed_data.get("cat")),
    num_dataset=torch.tensor(transformed_data.get("num"), dtype=torch.float32)
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. Initialize and train the FinDiff model
model = FinDiff(
    data_transformer=transformer,
    backbone_type="mlp",  # Try 'transformer' or 'unet'
    diffusion_total_steps=1000,
    num_epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

model.fit(dataloader)

# 4. Generate synthetic data
synthetic_df = model.sample(n_samples=5)
print(synthetic_df)
```
