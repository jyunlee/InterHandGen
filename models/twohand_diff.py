import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

### the embedding of diffusion timestep ###
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
    
class TwoHandDiff(nn.Module):
    def __init__(self, config):
        super(TwoHandDiff, self).__init__()
        
        con = config.model
        self.hid_dim, self.emd_dim, self.coords_dim, num_layers, n_head, dropout, self.skips = \
            con.hid_dim, con.emd_dim, con.coords_dim, \
                con.num_layer, con.n_head, con.dropout, con.skips
                
        self.n_layers = num_layers

        self.fc_input = nn.Linear(self.coords_dim, self.hid_dim)
        _fc_layers = []

        for i in range(num_layers):
            if i in self.skips: 
                _fc_layers.append(nn.Linear(self.hid_dim*2 + self.emd_dim*2, self.hid_dim))
            else:
                _fc_layers.append(nn.Linear(self.hid_dim + self.emd_dim*2, self.hid_dim))

        self.fc_layers = nn.ModuleList(_fc_layers)
        self.fc_output = nn.Linear(self.hid_dim, self.coords_dim)
        
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.emd_dim, self.hid_dim),
            torch.nn.Linear(self.hid_dim, self.emd_dim),
        ])

        self.cemb = nn.Module()
        self.cemb.dense = nn.ModuleList([
            torch.nn.Linear(self.coords_dim, self.hid_dim),
            torch.nn.Linear(self.hid_dim, self.emd_dim),
        ])

        self.iemb = nn.Module()
        self.iemb.dense = nn.ModuleList([
            torch.nn.Linear(self.coords_dim, self.hid_dim),
            torch.nn.Linear(self.hid_dim, self.emd_dim),
        ])

        trans_enc_layer = nn.TransformerEncoderLayer(d_model=self.emd_dim, nhead=4)
        self.trans_enc = nn.TransformerEncoder(trans_enc_layer, num_layers=2)
        self.trans_out = nn.Linear(self.emd_dim*3, self.hid_dim)
 

    def forward(self, x, t, cond):

        # timestep embedding
        temb = get_timestep_embedding(t, self.emd_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        cemb = self.cemb.dense[0](cond)
        cemb = nonlinearity(cemb)
        cemb = self.cemb.dense[1](cemb)
 
        iemb = self.iemb.dense[0](x)
        iemb = nonlinearity(iemb)
        iemb = self.iemb.dense[1](iemb)

        out = self.trans_enc(torch.cat([temb.unsqueeze(1), cemb.unsqueeze(1), iemb.unsqueeze(1)], 1))
        out = out.reshape((out.shape[0], -1))
        out = self.trans_out(out)

        init = self.fc_input(x)

        for i in range(self.n_layers):
            if i in self.skips:
                out = torch.cat([out, init, temb, cemb], -1)
            else:
                out = torch.cat([out, temb, cemb], -1)

            out = self.fc_layers[i](out)
            out = F.relu(out)

        out = self.fc_output(out)

        return out


