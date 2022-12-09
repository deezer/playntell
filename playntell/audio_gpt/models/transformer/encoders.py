import math

import torch
from torch import nn
from torch.nn import functional as F

from audio_gpt.models.transformer.attention import MultiHeadAttention
from audio_gpt.models.transformer.utils import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=768,
        d_k=64,
        d_v=64,
        h=12,
        d_ff=2048,
        dropout=0.1,
        identity_map_reordering=False,
        attention_module=None,
        attention_module_kwargs=None,
    ):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(
            d_model,
            d_k,
            d_v,
            h,
            dropout,
            identity_map_reordering=identity_map_reordering,
            attention_module=attention_module,
            attention_module_kwargs=attention_module_kwargs,
        )
        self.pwff = PositionWiseFeedForward(
            d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering
        )

    def forward(
        self, queries, keys, values, attention_mask=None, attention_weights=None
    ):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(
        self,
        N,
        padding_idx,
        d_model=768,
        d_k=64,
        d_v=64,
        h=12,
        d_ff=2048,
        dropout=0.1,
        identity_map_reordering=False,
        attention_module=None,
        attention_module_kwargs=None,
    ):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_k,
                    d_v,
                    h,
                    d_ff,
                    dropout,
                    identity_map_reordering=identity_map_reordering,
                    attention_module=attention_module,
                    attention_module_kwargs=attention_module_kwargs,
                )
                for _ in range(N)
            ]
        )
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (
            (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
        )  # (b_s, 1, 1, seq_len)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask


class TransformInputDimensionality(nn.Module):
    def __init__(self, source_dimensionality, target_dimensionality, dropout=0.1):
        super(TransformInputDimensionality, self).__init__()
        self.fc = nn.Linear(source_dimensionality, target_dimensionality)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(target_dimensionality)

    def forward(self, input):
        out = gelu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class VisualEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(VisualEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        # out = F.relu(self.fc(input))

        out = gelu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        return super(VisualEncoder, self).forward(
            out, attention_weights=attention_weights
        )


class DummyVisualEncoder(nn.Module):
    def __init__(self, padding_idx, d_in, d_model=768):
        super(DummyVisualEncoder, self).__init__()
        self.padding_idx = padding_idx
        self.d_in = d_in

        transformations = []
        for source_dimensionality in d_in:
            target_dimensionality = d_model
            transformations.append(
                TransformInputDimensionality(
                    source_dimensionality, target_dimensionality
                )
            )
        self.transformations = nn.ModuleList(transformations)

    def forward(self, input):
        chunks = torch.split(input, self.d_in, dim=2)

        inputs = []
        for chunk, func in zip(chunks, self.transformations):
            inputs.append(func(chunk))

        attention_masks = [
            (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
            for input in inputs
        ]

        inputs = torch.cat(inputs, axis=-1)
        attention_masks = torch.cat(attention_masks, axis=-1)

        return inputs, attention_masks
