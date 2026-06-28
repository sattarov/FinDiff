import os
from typing import List, Optional, Dict, Any, Set
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    LabelEncoder,
    QuantileTransformer
)
from torch.utils.data import Dataset


class DataTransformer:
    """
    Tabular encoder/decoder tailored for FinDiff usage.

    Notes:
    - No automatic dtype inference. The user must provide categorical_cols
      and numerical_cols (or they default to empty lists).
    - No imputation / no drop_cols handling.
    - Numeric scaling (standard / minmax / robust / yeo-johnson).
    - Categorical encoding (ordinal).
    - Optional label_col encoded via LabelEncoder.
    - fit / transform / inverse_transform / save / load.
    - categorical_cardinality_ stores cardinality per categorical column.
    """

    _SCALERS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "power": PowerTransformer,  # use method='yeo-johnson' by default
        "quantile": QuantileTransformer,
    }

    def __init__(
        self,
        numerical_scaler: str = "standard",
        categorical_encoder: str = "label_encoder",  # only 'label_encoder' supported
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ):
        """
        Initialize the DataTransformer.

        Args:
            numerical_scaler (str, optional): The scaler to use for numerical features. Defaults to "standard".
            categorical_encoder (str, optional): The encoder to use for categorical features. Defaults to "label_encoder".
            categorical_cols (List[str], optional): List of categorical column names. Defaults to None.
            numerical_cols (List[str], optional): List of numerical column names. Defaults to None.
        """

        # column lists
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []

        self.numerical_scaler_name = numerical_scaler
        self.categorical_encoder_name = categorical_encoder

        # placeholders to be created on fit
        self.num_scaler: Any = None
        self.cat_encoder: Optional[LabelEncoder] = None
        self.label_encoder: Optional[LabelEncoder] = None

        # metadata
        self.ordinal_category_mapping_: Dict[str, List[Any]] = {}
        self.categorical_cardinality_: Dict[str, int] = {}
        self.categorical_mapping_: Dict[str, Set[Any]] = {}
        self.label_classes_: Optional[List[Any]] = None
        self.label_cardinality_: Optional[int] = None
        self.fitted = False

    def _build_scaler(self):
        key = self.numerical_scaler_name.lower()
        if key not in self._SCALERS:
            raise ValueError(f"Unknown scaler '{self.numerical_scaler_name}'")
        if key == "power":
            return self._SCALERS[key](method="yeo-johnson")
        if key == "quantile":
            return self._SCALERS[key](output_distribution="normal", random_state=0)
        return self._SCALERS[key]()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataTransformer":
        """
        Fit the data transformer on the given data.

        Args:
            X (pd.DataFrame): The input tabular data containing categorical and numerical columns.
            y (pd.Series, optional): The target labels for conditional generation. Defaults to None.

        Returns:
            DataTransformer: The fitted transformer instance.
        """
        # scaler
        if self.numerical_cols:
            self.num_scaler = self._build_scaler()
            num_values = X[self.numerical_cols].values.astype(float)
            self.num_scaler.fit(num_values)

        # categorical encoder (only label_encoder supported)
        enc_name = self.categorical_encoder_name.lower()
        if enc_name != "label_encoder":
            raise ValueError("Only 'label_encoder' categorical_encoder is supported")

        if self.categorical_cols:
            # prepend categorical column names to every category value for LabelEncoder
            cat_values = self.prepend_colnames(X[self.categorical_cols])
            self.cat_encoder = LabelEncoder()
            self.cat_encoder.fit(cat_values.values.flatten())
            self.categorical_mapping_ = {
                col: set(cat_values[col]) for col in self.categorical_cols
            }
            self.categorical_cardinality_ = {
                col: len(set(cat_values[col])) for col in self.categorical_cols
            }
            self.embedding_mapping_ = {
                col: np.where(np.isin(self.cat_encoder.classes_, list(self.categorical_mapping_[col])))[0]
                for col in self.categorical_cols
            }
        else:
            self.categorical_cardinality_ = {}

        # label encoder (optional)
        if y is not None:
            self.label_encoder = LabelEncoder()
            # fit on values; LabelEncoder accepts 1d array-like
            self.label_encoder.fit(y.astype(object).to_numpy().ravel())
            self.label_classes_ = list(self.label_encoder.classes_)
            self.label_cardinality_ = len(self.label_classes_)

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, np.ndarray]:
        """
        Transform the input data using the fitted scalers and encoders.

        Args:
            X (pd.DataFrame): The input tabular data to transform.
            y (pd.Series, optional): The target labels to transform. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing transformed 'num', 'cat', and optionally 'label' arrays.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")
        if X is None:
            raise ValueError("X must be provided for transform()")
        # only operate on provided column lists
        outputs = {}

        # numeric
        if self.numerical_cols:
            num_df = X[self.numerical_cols].copy()
            num_arr = num_df.values.astype(float)
            num_scaled = self.num_scaler.transform(num_arr)
            outputs["num"] = num_scaled

        # categorical (ordinal)
        if self.categorical_cols:
            assert self.cat_encoder is not None
            cat_df = X[self.categorical_cols].copy()
            # prepend categorical column names to every category value for LabelEncoder
            # check if column names are already prepended to avoid double prepending in case of multiple transform calls
            if not all(cat_df[col].astype(str).str.startswith(f"{col}__").all() for col in self.categorical_cols):
                cat_df = self.prepend_colnames(cat_df)
            cat_enc = cat_df.apply(self.cat_encoder.transform).values # type: ignore            
            outputs["cat"] = cat_enc

        # label (optional) - use LabelEncoder; unseen labels mapped to -1
        if y is not None:
            assert self.label_encoder is not None, "Call fit() with labels before transform() when y is provided"
            label_vals = y.astype(object).to_numpy().ravel()
            label_encoded = self.label_encoder.transform(label_vals)
            # label_col_out = f"label__{self.label_col}"
            outputs["label"] = label_encoded

        return outputs

    def inverse_transform(self, X: dict) -> pd.DataFrame:
        """
        Reconstruct original DataFrame (only columns that were part of encoder).
        
        Args:
            X (dict): Dictionary containing 'num', 'cat', and optionally 'label' arrays,
                      usually produced by the `transform()` method.
                      
        Returns:
            pd.DataFrame: The inverse-transformed pandas DataFrame.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before inverse_transform()")

        num_df = pd.DataFrame(columns=self.numerical_cols)
        cat_df = pd.DataFrame(columns=self.categorical_cols)
        label_df = None

        # numeric inverse
        if self.numerical_cols and "num" in X:
            num_arr = X["num"]
            num_inv = self.num_scaler.inverse_transform(num_arr)
            num_df = pd.DataFrame(num_inv, columns=self.numerical_cols)

        # categorical inverse (ordinal)
        if self.categorical_cols and "cat" in X:
            assert self.cat_encoder is not None, "Call fit() with categorical columns before inverse_transform()"
            cat_nums = pd.DataFrame(X["cat"], columns=self.categorical_cols)
            cat_df = cat_nums.apply(self.cat_encoder.inverse_transform)
            # remove prepended column name from every category value
            for col in self.categorical_cols:
                # Vectorized string split is significantly faster than .apply(lambda)
                cat_df[col] = cat_df[col].astype(str).str.split("__", n=1).str[-1]

        # label inverse
        if self.label_encoder is not None and "label" in X:
            label_nums = X["label"]
            label_inv = self.label_encoder.inverse_transform(label_nums)
            label_df = pd.Series(label_inv, name="label")

        # combine categorical and numeric (and label if exists)
        dfs = []
        if self.categorical_cols and "cat" in X:
            dfs.append(cat_df)
        if self.numerical_cols and "num" in X:
            dfs.append(num_df)
        rec_df = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
        if label_df is not None:
            rec_df = pd.concat([rec_df, label_df], axis=1)
        return rec_df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, np.ndarray]:
        """
        Fit the transformer to the data and then transform it.

        Args:
            X (pd.DataFrame): The input tabular data.
            y (pd.Series, optional): The target labels. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing transformed arrays.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def save(self, path: str) -> None:
        """
        Save the fitted DataTransformer instance to disk.
        
        Args:
            path (str): The file path where the transformer should be saved.
        """
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "DataTransformer":
        """
        Load a fitted DataTransformer instance from disk.
        
        Args:
            path (str): The file path from which to load the transformer.
            
        Returns:
            DataTransformer: The loaded transformer instance.
        """
        return joblib.load(path)
    
    def prepend_colnames(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepend column names to categorical values for correct encoding by LabelEncoder.
        
        Args:
            df (pd.DataFrame): The input dataframe containing categorical columns.
            
        Returns:
            pd.DataFrame: A new dataframe with modified categorical values.
        """
        df_copy = df.copy()
        for col in self.categorical_cols:
            # Vectorized string concatenation
            df_copy[col] = f"{col}__" + df_copy[col].astype(str)
        return df_copy


class FinDiffDataset(Dataset):
    """PyTorch Dataset for FinDiff tabular data."""

    # define the class constructor
    def __init__(self, cat_dataset, num_dataset, labels=None):

        # set datasets
        self.cat_dataset = cat_dataset
        self.num_dataset = num_dataset
        self.labels = labels

    # define the length method
    def __len__(self):

        # returns the number of samples
        return len(self.num_dataset)

    # define the get item method
    def __getitem__(self, index):

        sample ={
            "cat": self.cat_dataset[index, :],
            "num": self.num_dataset[index, :]
        }
        if self.labels is not None:
            sample["label"] = self.labels[index]
      
        # return dataset and label batch
        return sample