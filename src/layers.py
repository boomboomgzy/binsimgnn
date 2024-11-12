"""Classes for SimGNN modules."""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention



# channel=node_embedding_dim+pe_dim
class GPS(torch.nn.Module):
    def __init__(self, channels: int, num_layers: int, heads: int,
                 attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        #self.pe_lin = Linear(8, pe_dim)
        #self.pe_norm = BatchNorm1d(8)
        self.edge_emb = Embedding(215, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=heads,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        #redraw_interval==0 每次调用都会重绘
        self.redraw_projection = RedrawProjection(self.convs,redraw_interval=0 if attn_type == 'performer' else None)
    
    def forward(self, x, edge_index, edge_attr, batch):


        #x_pe = self.pe_norm(pe)
        #x = torch.cat((x, self.pe_lin(x_pe)), dim=1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = global_add_pool(x, batch)
        return x


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            #redraw
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            print('projections has been redraw !')
            return
        self.num_last_redraw += 1



