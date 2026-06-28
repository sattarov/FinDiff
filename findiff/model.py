import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from findiff.diffusion import BaseDiffuser
from findiff.backbones import FinDiffSynthesizer
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn as nn
import torch.optim as optim

class FinDiff:
    def __init__(
            self,
            data_transformer,
            cat_decoding: str = 'distance', 
            cat_emb_dim: int = 2,
            timestep_emb_dim: int = 64,
            emb_learnable: bool = False, 
            diffusion_total_steps: int = 1000,
            diffusion_beta_start: float = 0.0001,
            diffusion_beta_end: float = 0.02,
            diffusion_scheduler: str = 'linear',
            diffusion_scheduler_scale: bool = False,
            device: str = 'cpu',
            num_epochs: int = 100,
            learning_rate: float = 1e-4,
            batch_size: int = 128,
            batch_size_sample: int | None = None,
            backbone_type: str = 'mlp',
            backbone_config: dict | None = None
        ):
        self.cat_decoding = cat_decoding
        self.learning_rate = learning_rate
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.batch_size_sample = batch_size_sample  # default sample batch size same as training batch size
        self.data_transformer = data_transformer
        self.cat_emb_dim = cat_emb_dim
        self.cat_cols_dim = len(data_transformer.categorical_cols)

        # init the synthesizer model
        self.synthesizer = FinDiffSynthesizer(
            num_cols_dim=len(data_transformer.numerical_cols),
            cat_vocab=data_transformer.categorical_mapping_,
            n_classes=data_transformer.label_cardinality_,
            cat_emb_dim=cat_emb_dim,
            time_embed_dim=timestep_emb_dim,
            embedding_learned=emb_learnable,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            cat_decoding=self.cat_decoding
        )

        # push model to compute device
        self.synthesizer = self.synthesizer.to(self.device)

        # init the diffuser
        self.diffuser = BaseDiffuser(
            total_steps=diffusion_total_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
            device=device,
            scheduler=diffusion_scheduler,
            scheduler_scale=diffusion_scheduler_scale
        )

        # int mean-squared-error loss
        if self.cat_decoding == 'logits':
            self.loss_fnc_cat = nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.loss_fnc = nn.MSELoss(reduction='none').to(self.device)

        # determine synthesizer model parameters
        parameters = filter(lambda p: p.requires_grad, self.synthesizer.parameters())

        # init the adam model optimizer
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        # init the cosine learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        # placeholder for training history
        self.training_history = {
            "train_loss": []
        }


    def fit(self, train_dataloader):
        """ Fit function for training the FinDiff model

        Args:
            train_dataloader (DataLoader): PyTorch DataLoader containing training data
        """

        # init the training progress bar
        pbar = tqdm(iterable=range(self.num_epochs), position=0, leave=True)

        # iterate over training epochs
        for epoch in pbar:

            loss = self.train_epoch(
                dataloader=train_dataloader,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
            )
                
            # determine mean training round loss
            pbar.set_description(f"Epoch {epoch + 1:04d} | Train Loss: {loss:.6f}")

            # save training history
            self.training_history["train_loss"].append(loss)

    def train_epoch(
            self,
            dataloader,
            optimizer,
            scheduler,
        ):
        """Training module for single epoch, update model parameters and return losses

        Args:
            dataloader (DataLoader): torch Dataloader
            optimizer (Optimizer): optimizer
            scheduler (LRScheduler): learning rate scheduler

        Returns:
            float: mean loss for the epoch
        """
        total_losses = []

        # iterate over distinct mini-batches
        for batch in dataloader:
            # set network in training mode
            self.synthesizer.train()
            # push batch to compute device
            batch_cat = batch["cat"].to(self.device)
            batch_num = batch["num"].to(self.device)
            if "label" in batch:
                batch_y = batch["label"].to(self.device)
            else:                
                batch_y = None

            # sample timestamps t
            timesteps = self.diffuser.sample_timesteps(n=batch_cat.shape[0])

            # get cat embeddings
            batch_cat_emb = self.synthesizer.embed_x_cat(x_cat=batch_cat)
            # concat cat & num
            batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

            # add noise
            batch_noise_t, noise_t = self.diffuser.add_gauss_noise(x_num=batch_cat_num, t=timesteps)

            # conduct forward encoder/decoder pass
            if self.cat_decoding == 'distance':
                predicted_noise = self.synthesizer(x=batch_noise_t, timesteps=timesteps, label=batch_y)

                # compute train loss
                train_losses = self.loss_fnc(
                    input=noise_t,
                    target=predicted_noise,
                ).sum(dim=1).mean()

            elif self.cat_decoding == 'logits':
                predicted_noise_num, cat_logits = self.synthesizer(x=batch_noise_t, timesteps=timesteps, label=batch_y)

                # compute num loss
                num_loss = self.loss_fnc(
                    input=noise_t,
                    target=predicted_noise_num,
                ).sum(dim=1)

                # compute cat loss
                cat_loss = 0
                for i in range(len(cat_logits)):
                    # get target cat column and adjust for min index of column in embedding mapping
                    cat_min_idx = list(self.data_transformer.embedding_mapping_.values())[i].min()
                    batch_cat_target = (batch_cat[:, i] - cat_min_idx).long()
                    cat_loss += self.loss_fnc_cat(cat_logits[i], batch_cat_target)
                
                # total loss
                train_losses = (num_loss + cat_loss).mean()

            else:
                raise ValueError(f"Unknown cat_decoding: {self.cat_decoding}")

            # reset encoder and decoder gradients
            optimizer.zero_grad()

            # run error back-propagation
            train_losses.backward()

            # optimize encoder and decoder parameters
            optimizer.step()

            # collect rec error losses
            total_losses.append(train_losses.detach().cpu().numpy())

        # average of rec errors
        total_losses_mean = np.mean(np.array(total_losses))

        # update learning rate according to the scheduler
        scheduler.step()

        return total_losses_mean

    @torch.no_grad()
    def sample(self, n_samples:int | None=None, label: torch.Tensor | None=None):
        """ Generation of samples. 
            For unconditional sampling use n_samples, for conditional sampling provide label.

        Args:
            n_samples (int, optional): number of samples to sample. Defaults to None.
            label (tensor, optional): list of labels for conditional sampling. Defaults to None.

        Raises:
            Exception: If neither n_samples nor label is provided.

        Returns:
            pd.DataFrame: decoded sample dataframe
        """

        if (n_samples is None) and (label is None):
            raise Exception("either n_samples or label needs to be given")

        if label is not None:
            n_samples = len(label)
            label = label.to(device=self.device)

        # logits for cat output when one-hot output is used        
        cat_logits = None
        # get dimension of encoded data
        dim_input = self.synthesizer.dim_input

        # ensure n_samples is available for sample generation
        assert n_samples is not None
        chunk_size: int = self.batch_size_sample if self.batch_size_sample is not None else n_samples
        assert chunk_size is not None

        # decide chunk size for batch-wise processing
        total_chunks = int(np.ceil(n_samples / chunk_size))

        decoded_list = []

        # Process samples in chunks to reduce peak memory usage.
        for chunk_idx, start in enumerate(range(0, n_samples, chunk_size)):
            cur_n = min(chunk_size, n_samples - start)

            # slice labels for current chunk if provided
            label_chunk = None
            if label is not None:
                label_chunk = label[start: start + cur_n]

            # init random noise for this chunk
            z = torch.randn((cur_n, dim_input), device=self.device)

            # run the reverse diffusion for this chunk
            pbar = tqdm(iterable=reversed(range(0, self.diffuser.total_steps)),
                        desc=f"SAMPLING chunk {start}-{start+cur_n-1}", leave=False)
            for i in pbar:
                pbar.set_description(f"SAMPLING STEP: {i:4d} (chunk ({chunk_idx+1}/{total_chunks}) {start}-{start+cur_n-1})")

                t = torch.full((cur_n,), i, dtype=torch.long, device=self.device)

                if self.cat_decoding == 'distance':
                    model_out = self.synthesizer(z.float(), t, label_chunk)
                elif self.cat_decoding == 'logits':
                    model_out, cat_logits = self.synthesizer(z.float(), t, label_chunk)
                else:
                    raise ValueError(f"Unknown cat_decoding: {self.cat_decoding}")

                z = self.diffuser.p_sample_gauss(model_out, z, t)

            # decode samples for the chunk and collect
            df_chunk = self.decode_sample(sample=z, cat_logits=cat_logits)
            decoded_list.append(df_chunk)

        # concatenate decoded chunks preserving order
        if len(decoded_list) > 0:
            x_0 = pd.concat(decoded_list, ignore_index=True)
        else:
            x_0 = pd.DataFrame()

        return x_0
    
    def decode_sample(
            self,
            sample,
            cat_logits=None,
        ):
        """ Decoding function for unscaling numeric attributes and inverse encoding of categorical attributes.
            Used once synthetic data is generated. 

        Args:
            sample (tensor): input samples for decoding
            cat_logits (tensor, optional): predicted categorical logits for decoding when one-hot encoding is used. Defaults to None.

        Returns:
            pandas DataFrame: decoded dataframe
        """
        cat_dim = self.cat_emb_dim * self.cat_cols_dim
        # split sample into numeric and categorical parts
        sample = sample.cpu().numpy()
        sample_num = sample[:, cat_dim:]
        sample_cat = sample[:, :cat_dim]

        # decode categorical embeddings to categorical codes
        if self.cat_decoding == 'distance':
            sample_cat = self.decode_cat_emb_distance(sample_cat)
        elif self.cat_decoding == 'logits':
            sample_cat = self.decode_cat_emb_logits(cat_logits)
        else:
            raise ValueError(f"Categorical decoding {self.cat_decoding} not supported.")
        
        # inverse transform to original data space
        sample_decoded = self.data_transformer.inverse_transform({"cat": sample_cat, "num": sample_num})

        return sample_decoded

    def decode_cat_emb_distance(
            self,
            sample_cat
        ):
        """ Decoding function for inverse encoding of categorical attributes from embeddings.
            Used once synthetic data is generated. 

        Args:
            sample_cat (np.ndarray): input categorical samples for decoding

        Returns:
            np.ndarray: decoded categorical data array
        """
        # get embedding lookup matrix
        embedding_lookup = self.synthesizer.get_x_cat_emb().cpu()
        # categorical attributes
        cat_attrs = self.data_transformer.embedding_mapping_.keys()
        # reshape back to batch_size * n_dim_cat * cat_emb_dim
        sample_cat = sample_cat.reshape(-1, len(cat_attrs), self.cat_emb_dim)
        # compute batch-wise calculation of distances for memory efficiency
        batch_size = self.batch_size
        n_samples = len(sample_cat)
        z_cat_list = []

        # iterate over generated categorical samples
        for i in range(0, n_samples, batch_size):
            # get batch of samples
            samples_cat_subset = sample_cat[i: i+batch_size]
            # get the closest distance based on the embeddings that belong to a column category
            distances = torch.cdist(x1=embedding_lookup, x2=torch.as_tensor(samples_cat_subset, dtype=torch.float32))
            z_cat = []
            for attr_idx, attr_name in enumerate(cat_attrs):
                # get vocab indices for attribute
                attr_emb_idx = list(self.data_transformer.embedding_mapping_[attr_name])
                # get distances for attribute
                attr_distances = distances[:, attr_emb_idx, attr_idx]
                # get nearest embedding index
                _, nearest_idx = torch.min(attr_distances, dim=1)
                # convert to numpy
                nearest_idx = nearest_idx.cpu().numpy()
                # map emb indices back to column indices
                z_cat.append(np.array(attr_emb_idx)[nearest_idx])

            z_cat = np.array(z_cat).T
            z_cat_list.append(z_cat)
        z_cat = np.vstack(z_cat_list)

        return z_cat

    def decode_cat_emb_logits(
            self,
            cat_logits
        ):
        """
        Decoding function for inverse encoding of categorical attributes from logits.

        Args:
            cat_logits (list of torch.Tensor): List of predicted categorical logits.

        Returns:
            np.ndarray: Decoded categorical codes.
        """
        z_cat = []

        for cat_idx, cat_logit in enumerate(cat_logits):
            cat_logit = torch.argmax(cat_logit, dim=1)
            # add min index of cat column
            cat_min_idx = list(self.data_transformer.embedding_mapping_.values())[cat_idx].min()
            cat_logit += cat_min_idx
            z_cat.append(cat_logit)

        z_cat = torch.stack(z_cat, dim=1)
        z_cat = z_cat.cpu().numpy()

        return z_cat