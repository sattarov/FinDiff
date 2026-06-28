# FinDiff

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sattarov/FinDiff/blob/master/examples/extended.ipynb)

Implementation of the paper: **"FinDiff: Diffusion Models for Financial Tabular Data Generation"** [link](https://dl.acm.org/doi/abs/10.1145/3604237.3626876).

FinDiff is a PyTorch-based framework for generating synthetic tabular data, specifically designed to handle the complexities of financial datasets (e.g., mixtures of continuous and categorical variables). It leverages diffusion models with swappable neural network backbones to synthesize realistic records.

## Features

- **Efficient Categorical Handling**: Learnable embeddings for categorical data, with support for decoding via distance-based matching or direct logit prediction.
- **Conditional Generation**: Support for label-conditioned synthetic data generation using Classifier-Free Guidance.
- **Multiple Neural Backbones**: Choose between MLP, Transformer, and U-Net backbones for the diffusion process.
- **Robust Data Transformation**: Built-in `DataTransformer` handles standardizing numerical features (Standard, MinMax, Robust, Power, Quantile) and encoding categorical features.
- **Customizable Diffusion Schedulers**: Configurable noise schedules including linear, quadratic, sigmoid, and exponential.

## Examples

You can explore FinDiff's capabilities using our example notebooks on Google Colab:

- [Compact Example](./examples/compact.ipynb): A quick-start guide to get you up and running (examples/compact.ipynb).
  
- [Extended Example](./examples/extended.ipynb): A deep dive into the architecture and mathematics of the model (examples/extended.ipynb).


## Architecture

The framework is highly modular and broken down into four core components:

1. **The Orchestrator (`findiff/model.py`)**
   - **`FinDiff`**: The main user-facing class. It acts as the central orchestrator that wires together data transformations, diffusion mathematics, and the neural network. It handles the complete lifecycle, including the training loop (`fit()`) and synthetic data generation (`sample()`).

2. **Data Processing Pipeline (`findiff/data.py`)**
   - **`DataTransformer`**: A robust preprocessing pipeline built specifically for tabular data. It scales numerical features and encodes categorical variables. Importantly, it manages the `inverse_transform` to accurately convert raw diffusion outputs back into a readable pandas DataFrame.
   - **`FinDiffDataset`**: A standard PyTorch dataset wrapper to feed transformed tensors into dataloaders.

3. **Neural Network Backbones (`findiff/backbones.py`)**
   - **`FinDiffSynthesizer`**: The core predictive model that takes numerical data, embedded categorical data, and time-step embeddings to predict the noise during the reverse diffusion step.
   - **Interchangeable Architectures**: The framework implements three separate architectures that can be swapped based on data complexity:
     - `MLPBackbone`: Uses residual blocks for standard, lightweight tabular generation.
     - `TransformerBackbone`: An attention-based tabular model.
     - `UNetBackbone`: Adapts U-Net principles (typically used in image generation) for tabular representations.

4. **Diffusion Mathematics (`findiff/diffusion.py`)**
   - **`BaseDiffuser`**: Handles the underlying math for the diffusion process. It manages noise schedulers (Linear, Quadratic, Sigmoid, Exponential) and computes the parameters required to gradually add noise during the forward pass and iteratively denoise during generation.

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

## Citation

If this project helped your research, please consider giving it a star and citing it in your work. It helps keep the project alive.

```bibtex
@inproceedings{sattarov2023findiff,
  title={Findiff: Diffusion models for financial tabular data generation},
  author={Sattarov, Timur and Schreyer, Marco and Borth, Damian},
  booktitle={Proceedings of the Fourth ACM International Conference on AI in Finance},
  pages={64--72},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


