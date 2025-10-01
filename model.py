import torch
import torch.nn as nn
import math


class LAT_Module(nn.Module):
    """
    Local Self-Attention (LAT) Module.
    Applies self-attention within non-overlapping windows of patches.
    """

    def __init__(self, hidden_dim, num_heads, window_size, dropout=0.1):
        super(LAT_Module, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()

        remainder = num_patches % self.window_size
        padding = 0
        if remainder != 0:
            padding = self.window_size - remainder
            pad_tensor = x[:, -padding:, :].clone()
            x = torch.cat([x, pad_tensor], dim=1)

        num_windows = x.size(1) // self.window_size

        windows = x.view(batch_size, num_windows, self.window_size, self.hidden_dim)
        windows = windows.contiguous().view(batch_size * num_windows, self.window_size, self.hidden_dim)

        attn_out, _ = self.attention(windows, windows, windows)

        attn_out = attn_out.view(batch_size, num_windows, self.window_size, self.hidden_dim)
        attn_out = attn_out.view(batch_size, num_windows * self.window_size, self.hidden_dim)

        if padding != 0:
            attn_out = attn_out[:, :num_patches, :]

        return attn_out


class IDCformer(nn.Module):
    """
    The main IDCformer model architecture.
    """

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, seq_len, patch_size=16, window_size=6):
        super(IDCformer, self).__init__()

        self.patch_size = patch_size
        self.seq_len = seq_len
        self.num_patches = math.ceil(seq_len / patch_size)
        self.hidden_dim = hidden_dim

        self.patch_embed = nn.Linear(input_dim * patch_size, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.linear_layer = nn.Linear(seq_len, seq_len)
        self.tse_module = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=48, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=16, padding='same')
        )
        self.lat_module = LAT_Module(hidden_dim, num_heads, window_size)
        self.output_layer = nn.Linear(hidden_dim * self.num_patches, output_dim)

    def forward(self, x):
        linear_out = self.linear_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        tse_out = self.tse_module(x.permute(0, 2, 1)).permute(0, 2, 1)

        features_for_patching = linear_out + tse_out

        remainder = features_for_patching.size(1) % self.patch_size
        if remainder != 0:
            padding_size = self.patch_size - remainder
            pad_tensor = features_for_patching[:, -padding_size:, :].clone()
            features_for_patching = torch.cat([features_for_patching, pad_tensor], dim=1)

        current_num_patches = features_for_patching.size(1) // self.patch_size

        patches = features_for_patching.unfold(1, self.patch_size, self.patch_size).contiguous()
        patches = patches.view(features_for_patching.size(0), current_num_patches, -1)
        embedded_patches = self.patch_embed(patches)

        global_features = self.transformer_encoder(embedded_patches)
        local_features = self.lat_module(global_features)
        combined_features = global_features + local_features

        combined_features_flat = combined_features.reshape(combined_features.size(0), -1)
        output = self.output_layer(combined_features_flat)

        return output