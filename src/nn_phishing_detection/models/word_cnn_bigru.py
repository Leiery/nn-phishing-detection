"""
Word-level CNN-BiGRU model for phishing detection.

This module implements a neural network architecture that combines convolutional
layers for local feature extraction with bidirectional GRU layers for sequence
modeling. Designed for word-level phishing email classification.

Author: David Schatz <schatz@cl.uni-heidelberg.de>
"""
#ruff: noqa: E501
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpatialDropout1D(nn.Module):
    """
    Spatial dropout layer for 1D sequences.

    Unlike standard dropout which randomly sets individual elements to zero,
    spatial dropout sets entire feature channels to zero. This encourages
    the model to learn representations that don't rely on specific channels.

    Parameters
    ----------
    drop_rate : float
        Probability of dropping each channel (0.0 to 1.0).

    Notes
    ----------
    Input shape: (batch_size, channels, sequence_length)
    Output shape: Same as input

    During training, randomly selected channels are set to zero and remaining
    channels are scaled by 1/(1-drop_rate) to maintain expected values.
    """

    def __init__(self, drop_rate: float):
        """
        Initialize SpatialDropout1D layer.

        Parameters
        ----------
        drop_rate : float
            Dropout probability for each channel.
        """
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial dropout to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor with same shape as input, with random channels dropped.
        """
        if not self.training or self.drop_rate == 0:
            return x

        # x shape: (batch, length, channels)
        # We want to drop entire channels
        noise_shape = (x.shape[0], 1, x.shape[2])
        keep_prob = 1 - self.drop_rate
        mask = torch.bernoulli(torch.full(noise_shape, keep_prob, device=x.device))
        return x * mask / keep_prob


class ConvBlock(nn.Module):
    """
    1D Convolutional block with batch normalization and activation.

    Applies 1D convolution, batch normalization, and LeakyReLU activation
    with optional residual connections for improved gradient flow in deep networks.

    Parameters
    ----------
    in_channels : int
        Number of input channels (embedding dimension or previous layer output).
    out_channels : int
        Number of output channels (CNN filter count).
    kernel_size : int, default=3
        Size of the convolutional kernel (word context window).
    use_residual : bool, default=False
        Whether to add residual connection (only if in_channels == out_channels).

    Attributes
    ----------
    conv : nn.Conv1d
        1D convolutional layer with appropriate padding.
    bn : nn.BatchNorm1d
        Batch normalization layer.
    activation : nn.LeakyReLU
        LeakyReLU activation with negative slope 0.01.
    trim_output : bool
        Whether to trim output for even kernel sizes.

    Notes
    -----
    Handles both even and odd kernel sizes with appropriate padding.
    Even kernels use asymmetric padding and output trimming.
    Residual connections are only applied when channel dimensions match.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, use_residual: bool = False):
        """
        Initialize ConvBlock layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, default=3
            Convolutional kernel size.
        use_residual : bool, default=False
            Enable residual connections if dimensions match.
        """
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)

        if kernel_size % 2 == 0:
            # Even kernel: use asymmetric padding
            self.conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.trim_output = True
        else:
            # Odd kernel: standard symmetric padding
            self.conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.trim_output = False

        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length).

        Notes
        -----
        Applies convolution → batch normalization → LeakyReLU activation.
        Adds residual connection if use_residual=True and dimensions match.
        Automatically handles output trimming for even kernel sizes.
        """
        # x shape: (batch, channels, length)
        identity = x if self.use_residual else None

        out = self.conv(x)

        # Trim output for even kernels to match input size
        if self.trim_output and out.size(2) > x.size(2):
            out = out[:, :, :x.size(2)]

        out = self.bn(out)
        out = self.activation(out)

        # Residual connection
        if self.use_residual and identity is not None and identity.shape == out.shape:
            out = out + identity

        return out


class WordCNNBiGRU(nn.Module):
    """
    Word-level CNN-BiGRU model for phishing email classification.

    Combines convolutional neural networks for local feature extraction with
    bidirectional GRU for sequence modeling. Supports pretrained GloVe embeddings
    and implements spatial dropout, residual connections, and dual pooling.

    Architecture:
    1. Word embeddings (with GloVe pretrained)
    2. Spatial dropout on embeddings
    3. Multiple 1D CNN layers (capture local word patterns/phrases)
    4. Bidirectional GRU (capture long-range dependencies)
    5. Global max + mean pooling
    6. Classification head with dropout

    Parameters
    ----------
    vocab_size : int
        Size of vocabulary for embedding layer.
    embedding_dim : int, default=300
        Dimension of word embeddings (300 for GloVe).
    dropout : float, default=0.3
        Dropout probability for regularization.
    cnn_channels : int, default=300
        Number of CNN filter channels.
    conv_layers : int, default=3
        Number of convolutional layers.
    kernel_size : int, default=3
        Convolutional kernel size (word context window).
    gru_hidden : int, default=128
        Hidden size for GRU layer.
    gru_layers : int, default=1
        Number of GRU layers.
    num_classes : int, default=2
        Number of output classes (2 for binary classification).
    pretrained_embeddings : torch.Tensor or None, default=None
        Pretrained embedding weights (e.g., GloVe). If None, random initialization.

    Attributes
    ----------
    embedding : nn.Embedding
        Word embedding layer with padding_idx=0.
    embedding_dropout : SpatialDropout1D
        Spatial dropout applied to embeddings.
    conv_layers : nn.ModuleList
        List of ConvBlock modules for feature extraction.
    gru : nn.GRU
        Bidirectional GRU for sequence modeling.
    classifier : nn.Sequential
        Classification head with dropout and linear layers.

    Notes
    -----
    Combines global max pooling and global mean pooling for rich representations.
    Pretrained embeddings are fine-tunable (freeze=False).
    Gradient clipping should be applied during training via clip_gradients().
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,  # Common for GloVe
        dropout: float = 0.3,
        cnn_channels: int = 300,
        conv_layers: int = 3,
        kernel_size: int = 3,  # 3-word context window
        gru_hidden: int = 128,
        gru_layers: int = 1,
        num_classes: int = 2,  # Binary: legitimate vs malicious
        pretrained_embeddings: torch.Tensor | None = None
    ):
        """
        Initialize WordCNNBiGRU model.

        Parameters
        ----------
        vocab_size : int
            Vocabulary size.
        embedding_dim : int, default=300
            Embedding dimension.
        dropout : float, default=0.3
            Dropout probability.
        cnn_channels : int, default=300
            CNN channels.
        conv_layers : int, default=3
            Number of convolutional layers.
        kernel_size : int, default=3
            CNN kernel size.
        gru_hidden : int, default=128
            GRU hidden size.
        gru_layers : int, default=1
            Number of GRU layers.
        num_classes : int, default=2
            Number of output classes.
        pretrained_embeddings : torch.Tensor or None, default=None
            Pretrained embeddings (e.g., GloVe).
        """
        super().__init__()

        # Word embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,  # Allow fine-tuning
                padding_idx=0
            )
            embedding_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Spatial dropout on embeddings (prevents overfitting)
        self.embedding_dropout = SpatialDropout1D(dropout)

        # CNN layers (capture local word patterns)
        self.conv_layers = nn.ModuleList()
        in_channels = embedding_dim

        for i in range(conv_layers):
            self.conv_layers.append(
                ConvBlock(
                    in_channels if i == 0 else cnn_channels,
                    cnn_channels,
                    kernel_size,
                    use_residual=(i > 0)  # Residual connections after first layer
                )
            )

        # Bidirectional GRU (capture sequential patterns)
        self.gru = nn.GRU(
            cnn_channels,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        # Pooling combines max and mean for rich representation
        gru_output_size = gru_hidden * 2  # Bidirectional
        pooled_size = gru_output_size * 2  # Max + mean pooling

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_size, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through WordCNNBiGRU model.

        Processes word sequences through embeddings, CNN layers, bidirectional GRU,
        dual pooling, and classification head to produce class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input word indices of shape (batch_size, seq_len).
        lengths : torch.Tensor
            Actual sequence lengths of shape (batch_size,) for packed sequences.

        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes).

        Notes
        -----
        Applies masked pooling to handle padded sequences correctly.
        Combines max and mean pooling for richer feature representation.
        """
        batch_size, seq_len = x.shape

        # Word embeddings
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Spatial dropout on embeddings
        x = self.embedding_dropout(x)

        # CNN layers expect (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)

        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_channels)

        # Pack sequences for efficient processing
        x_packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Bidirectional GRU
        gru_out, _ = self.gru(x_packed)

        # Unpack sequences
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
        # (batch, seq_len, gru_hidden * 2)

        # Global pooling (combine max and mean across time dimension)
        # Create mask for proper pooling
        mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)

        # Masked pooling
        gru_out_masked = gru_out * mask.float()

        # Global max pooling
        max_pool = torch.max(gru_out_masked, dim=1)[0]  # (batch, gru_hidden * 2)

        # Global mean pooling
        mean_pool = torch.sum(gru_out_masked, dim=1) / lengths.unsqueeze(1).float()

        # Concatenate max and mean pooling
        pooled = torch.cat([max_pool, mean_pool], dim=1)  # (batch, gru_hidden * 4)

        # Classification
        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits

    def clip_gradients(self, max_norm: float = 1.0) -> None:
        """
        Clip gradients by global norm to prevent exploding gradients.

        Parameters
        ----------
        max_norm : float, default=1.0
            Maximum allowed gradient norm. Gradients are scaled down if their
            global norm exceeds this value.

        Returns
        -------
        None

        Notes
        -----
        Should be called after loss.backward() and before optimizer.step().
        Uses PyTorch's clip_grad_norm_ for global gradient norm clipping.
        """
        nn.utils.clip_grad_norm_(self.parameters(), max_norm)