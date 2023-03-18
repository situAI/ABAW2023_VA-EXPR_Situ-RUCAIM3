import torch
import torch.nn as nn
import numpy as np

from utils.registery import MODEL_REGISTRY


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table)


class TransformerEncoder(nn.Module):
    def __init__(self, inc, nheads, feedforward_dim, nlayers, dropout):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=inc,
            nhead=nheads,
            dim_feedforward=feedforward_dim,
            dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x):
        out = self.transformer_encoder(x)

        return out


def Regressor(inc_dim, out_dim, dims_list=[512, 256], dropout=0.3, act=nn.GELU(), has_tanh=True):
    module_list = list()
    module_list.append(nn.Linear(inc_dim, dims_list[0]))
    module_list.append(act)
    if dropout != None:
        module_list.append(nn.Dropout(dropout))
    for i in range(len(dims_list) - 1):
        module_list.append(nn.Linear(dims_list[i], dims_list[i + 1]))
        module_list.append(act)
        if dropout != None:
            module_list.append(nn.Dropout(dropout))

    module_list.append(nn.Linear(dims_list[-1], out_dim))
    if has_tanh:
        module_list.append(nn.Tanh())
    module = nn.Sequential(*module_list)

    return module


@MODEL_REGISTRY.register()
class BERT(nn.Module):
    def __init__(self,
                 input_dim,
                 feedforward_dim,
                 affine_dim,
                 nheads,
                 nlayers,
                 dropout,
                 use_pe,
                 seq_len,
                 head_dropout,
                 head_dims,
                 out_dim,
                 task):

        super().__init__()

        self.input_dim = input_dim
        inc = input_dim
        self.feedforward_dim = feedforward_dim
        self.affine_dim = affine_dim
        self.task = task
        if self.affine_dim != None:
            self.affine = nn.Linear(input_dim, affine_dim, bias=False) 
            inc = affine_dim

        self.use_pe = use_pe
        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc), freeze=True)

        self.transformer_encoder = TransformerEncoder(inc=inc, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers, dropout=dropout)

        if self.task == 'va':
            self.v_head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.a_head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
        else:
            self.head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)

    def forward(self, x):
        seq_len, bs, _ = x.shape
        if self.affine_dim != None:
            x = self.affine(x)
        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            x = x + position_embeddings

        out = self.transformer_encoder(x)

        if self.task == 'va':
            v_out = self.v_head(out)
            a_out = self.a_head(out)
            out = torch.cat([v_out, a_out], dim=-1)
        else:
            out = self.head(out)

        return out
