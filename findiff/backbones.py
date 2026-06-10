import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal time embedding module.
    
    Args:
        dim (int): The dimension of the sinusoidal embeddings. Must be even.
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Sinusoidal dimension must be even"
        self.dim = dim

    def forward(self, time):
        """
        Forward pass for sinusoidal position embeddings.
        
        Args:
            time (torch.Tensor): Timesteps of shape [batch_size].
            
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape [batch_size, dim].
        """
        # time shape: [batch_size]
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # embeddings shape: [batch_size, dim]
        return embeddings

class ResidualBlock(nn.Module):
    """A simple residual block for the MLP."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        return identity + out
    

class MLPBackbone(nn.Module):
    """
    MLP Backbone.
    Accepts concatenated x_t and a pre-computed conditioning embedding.
    
    Args:
        num_features (int): The dimension of the input tabular features.
        cond_dim (int): The dimension of the conditioning embedding.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
        num_blocks (int, optional): Number of residual blocks. Defaults to 4.
    """
    def __init__(self, num_features, cond_dim, hidden_dim=256, num_blocks=4):
        super().__init__()
        
        # Input projection now takes features + condition embedding
        self.input_proj = nn.Linear(num_features + cond_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, num_features)

    def forward(self, x, cond_emb):
        # x shape: [batch_size, num_features]
        # cond_emb shape: [batch_size, cond_dim]
        
        inputs = torch.cat([x, cond_emb], dim=1) # [B, num_features + cond_dim]
        
        h = self.input_proj(inputs)
        for block in self.blocks:
            h = block(h)
            
        output = self.output_proj(h)
        return output

# ---

class TransformerBackbone(nn.Module):
    """
    Transformer Backbone.
    Accepts x_t and a pre-computed conditioning embedding.
    
    Args:
        num_features (int): The dimension of the input tabular features.
        cond_dim (int): The dimension of the conditioning embedding.
        d_model (int, optional): The dimension of the transformer model. Defaults to 256.
        n_head (int, optional): Number of attention heads. Defaults to 8.
        num_layers (int, optional): Number of transformer encoder layers. Defaults to 6.
        dim_feedforward (int, optional): Dimension of the feedforward network model. Defaults to 1024.
    """
    def __init__(self, num_features, cond_dim, d_model=256, n_head=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        # Project conditioning embedding to d_model
        self.cond_proj = nn.Linear(cond_dim, d_model)
        
        self.feature_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(num_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, cond_emb):
        # x shape: [batch_size, num_features]
        # cond_emb shape: [batch_size, cond_dim]
        
        # 1. Project condition to d_model
        cond_emb_proj = self.cond_proj(cond_emb) # [batch_size, d_model]
        
        # 2. Reshape and embed features
        x = x.unsqueeze(-1)
        x_emb = self.feature_embed(x) # [batch_size, num_features, d_model]
        
        # 3. Add positional embedding
        pos_emb = self.pos_embed.weight.unsqueeze(0)
        x_emb = x_emb + pos_emb
        
        # 4. Add condition embedding (broadcasted)
        x_with_cond = x_emb + cond_emb_proj.unsqueeze(1)
        
        # 5. Pass through transformer
        transformer_out = self.transformer_encoder(x_with_cond)
        
        # 6. Project back to noise
        output = self.output_proj(transformer_out)
        
        return output.squeeze(-1)

class UNetBackbone(nn.Module):
    """
    U-Net backbone.
    - Conditioning added as per-layer linear projection and summed pre-activation.
    
    Args:
        num_features (int): The dimension of the input tabular features.
        cond_dim (int): The dimension of the conditioning embedding.
        hidden_dims (list of int, optional): Dimensions of the hidden layers. Defaults to [256, 512].
        activation (callable, optional): Activation function to use. Defaults to nn.SiLU.
    """
    def __init__(self, num_features, cond_dim, hidden_dims=[256, 512], activation=nn.SiLU):
        super().__init__()
        self.num_features = num_features
        self.hidden_dims = list(hidden_dims)

        # --- Encoder (Down Path) ---
        self.down_blocks = nn.ModuleList()
        self.cond_projs = nn.ModuleList()

        # in/out dims for down path: [num_features -> h0, h0 -> h1, ...]
        in_dim = num_features
        for out_dim in self.hidden_dims:
            self.down_blocks.append(nn.Linear(in_dim, out_dim))
            self.cond_projs.append(nn.Linear(cond_dim, out_dim))
            in_dim = out_dim

        # --- Bottleneck ---
        bot_dim = self.hidden_dims[-1] if self.hidden_dims else num_features
        self.bottleneck = nn.Linear(bot_dim, bot_dim)
        self.cond_projs.append(nn.Linear(cond_dim, bot_dim))

        # --- Decoder (Up Path) ---
        # Build sequentially so in_dim matches the runtime concatenation (h + skip)
        self.up_blocks = nn.ModuleList()
        h_dim = bot_dim
        for i in reversed(range(len(self.hidden_dims))):
            skip_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i-1] if i > 0 else num_features
            in_dim = h_dim + skip_dim
            self.up_blocks.append(nn.Linear(in_dim, out_dim))
            self.cond_projs.append(nn.Linear(cond_dim, out_dim))
            h_dim = out_dim  # will be input 'h' width for the next up block

        self.activation = activation()

    def forward(self, x, cond_emb):
        """
        Forward pass for the U-Net backbone.
        
        Args:
            x (torch.Tensor): Input noisy data of shape [batch_size, num_features].
            cond_emb (torch.Tensor): Conditioning embedding of shape [batch_size, cond_dim].
            
        Returns:
            torch.Tensor: Denoised data (or noise prediction) of shape [batch_size, num_features].
        """
        skips = []
        h = x
        cond_iter = iter(self.cond_projs)

        # --- Down Path ---
        for block in self.down_blocks:
            c = next(cond_iter)(cond_emb)      # [B, out_dim]
            h = self.activation(block(h) + c)  # [B, out_dim]
            skips.append(h)                    # store activated output

        # --- Bottleneck ---
        c = next(cond_iter)(cond_emb)          # [B, bot_dim]
        h = self.activation(self.bottleneck(h) + c)

        # --- Up Path ---
        for block in self.up_blocks:
            c = next(cond_iter)(cond_emb)      # [B, out_dim]
            skip = skips.pop()                 # matches decoder depth
            h = torch.cat([h, skip], dim=1)    # [B, in_dim]
            h = self.activation(block(h) + c)  # [B, out_dim]

        return h

class FinDiffSynthesizer(nn.Module):
    """
    A conditional diffusion model for tabular data with swappable backbones.
    
    Args:
        num_cols_dim (int): The number of numerical features in the tabular data.
        cat_vocab (dict): Dictionary mapping categorical column names to their unique token indices.
        cat_emb_dim (int, optional): Dimension for categorical embeddings. Defaults to 2.
        n_classes (int | None, optional): The number of classes for conditional generation. Defaults to None.
        embedding_learned (bool, optional): Whether the categorical embeddings are learnable. Defaults to False.
        time_embed_dim (int | None, optional): The dimension for time and label embeddings. Defaults to 64.
        backbone_type (str, optional): The neural network backbone to use ('mlp', 'transformer', or 'unet'). Defaults to 'mlp'.
        backbone_config (dict | None, optional): A dictionary of hyperparameters for the chosen backbone. Defaults to None.
        cat_decoding (str, optional): The method for categorical decoding ('distance' or 'logits'). Defaults to 'distance'.
    """
    def __init__(
            self, 
            num_cols_dim: int, 
            cat_vocab: dict,
            cat_emb_dim: int = 2,
            n_classes: int | None = None, 
            embedding_learned: bool = False,
            time_embed_dim: int | None = None,
            backbone_type: str = 'mlp',            
            backbone_config: dict | None = None,
            cat_decoding: str = 'distance'
        ):
        super().__init__()

        self.cat_decoding = cat_decoding
        self.num_cols_dim = num_cols_dim
        n_cat_tokens = sum(len(tokens) for tokens in cat_vocab.values())
        self.dim_input = num_cols_dim + len(cat_vocab) * cat_emb_dim

        if backbone_config is None:
            backbone_config = {}

        if time_embed_dim is None:
            time_embed_dim = 64

        # --- 1. Conditioning Embeddings ---
        
        # Time embedding
        self.time_embed_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU()
        )
        # Label embedding
        if n_classes is not None:
            self.label_embed = nn.Embedding(n_classes, time_embed_dim)
        # Categorical embedding
        self.x_cat_emb = nn.Embedding(num_embeddings=n_cat_tokens, embedding_dim=cat_emb_dim)
        self.x_cat_emb.weight.requires_grad = embedding_learned
        if cat_decoding == 'logits':
            self.x_cat_emb.weight.requires_grad = True
        
        # Unconditional embedding (for Classifier-Free Guidance)
        # We learn this as a parameter
        self.uncond_embed = nn.Parameter(torch.randn(1, time_embed_dim))

        # --- 2. Backbone Switch ---
        self.backbone_type = backbone_type
        
        if backbone_type == 'mlp':
            self.backbone = MLPBackbone(
                num_features=self.dim_input,
                cond_dim=time_embed_dim,
                hidden_dim=backbone_config.get('mlp_hidden_dim', 256),
                num_blocks=backbone_config.get('mlp_num_residual_blocks', 2)
            )
        elif backbone_type == 'transformer':
            self.backbone = TransformerBackbone(
                num_features=self.dim_input,
                cond_dim=time_embed_dim,
                d_model=backbone_config.get('d_model', 1024),
                n_head=backbone_config.get('transformer_n_head', 2),
                num_layers=backbone_config.get('transformer_num_layers', 2),
                dim_feedforward=backbone_config.get('transformer_dim_feedforward', 4096)
            )
        elif backbone_type == 'unet':
            self.backbone = UNetBackbone(
                num_features=self.dim_input,
                cond_dim=time_embed_dim,
                hidden_dims=backbone_config.get('unet_hidden_dims', [256, 512])
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")
    

        if self.cat_decoding == 'logits':
            # self.cat_head = nn.Linear(self.dim_input, cond_embed_dim)
            self.cat_head = nn.Sequential(
                nn.Linear(self.dim_input, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
            self.x_cat_logits = nn.ModuleList(
                [nn.Linear(time_embed_dim, len(tokens)) for tokens in cat_vocab.values()]
            )            

    def get_x_cat_emb(self):
        """ Extract embedding vectors 

        Returns:
            tensor: embedding vectors
        """
        return self.x_cat_emb.weight.data

    def embed_x_cat(self, x_cat):
        """ Perform embedding mapping for categorical attributes

        Args:
            x_cat (tensor): categorical tokens

        Returns:
            tensor: embeddings
        """

        # perform embedding mapping and then reshape
        x_cat_emb = self.x_cat_emb(x_cat)
        x_cat_emb = x_cat_emb.view(-1, x_cat_emb.shape[1] * x_cat_emb.shape[2])
        return x_cat_emb
    
    def forward(self, x, timesteps, label=None):
        """
        Forward pass of the denoising model.
        
        Args:
            x (torch.Tensor): The noisy input data (x_t) of shape [batch_size, num_features]
            timesteps (torch.Tensor): The timesteps of shape [batch_size]
            label (torch.Tensor, optional): 
                The conditional labels of shape [batch_size].
                If None, uses unconditional embedding (for CFG).
                
        Returns:
            torch.Tensor or tuple: 
                If cat_decoding is 'distance', returns predicted noise of shape [batch_size, num_features].
                If cat_decoding is 'logits', returns a tuple containing predicted noise and categorical logits.
        """
        
        # --- 1. Compute Conditioning Embedding ---
        
        # [batch_size, cond_embed_dim]
        t_emb = self.time_embed_mlp(timesteps)
        
        # [batch_size, cond_embed_dim]
        if label is not None:
            l_emb = self.label_embed(label)
        else:
            # Use unconditional embedding for all samples in batch
            l_emb = self.uncond_embed.repeat(x.shape[0], 1)
            
        # Combine embeddings
        cond_emb = t_emb + l_emb

        # --- 2. Pass to Backbone ---
        
        # The backbone predicts the noise
        x_pred = self.backbone(x, cond_emb)

        if self.cat_decoding == 'distance':
            return x_pred
        elif self.cat_decoding == 'logits':
            cat_head = self.cat_head(x_pred)
            cat_head += t_emb

            # get logits
            cat_logits = [cat_logit(cat_head) for cat_logit in self.x_cat_logits]
            return x_pred, cat_logits
        else:
            raise ValueError(f"Unknown cat_decoding: {self.cat_decoding}")