import torch
import torch.nn as nn
from .bert_model import Regressor
from utils.registery import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class BiGRUEncoder(nn.Module):
    def __init__(self,
                 input_dim=1024,
                 affine_dim=1024,
                 hidden_size=1024,
                 num_layers=4,
                 bias=True,
                 batch_first=True,
                 dropout=0.3,
                 bidirectional=True,
                 task='va',
                 out_dim=1,
                 head_dropout=0.1,
                 head_dims=[512, 256]):
        super().__init__()
        self.affine_dim = affine_dim
        if affine_dim:
            self.affine = nn.Linear(input_dim, affine_dim)
            inc = affine_dim
        else:
            inc = input_dim
        self.task = task
        self.gru_encoder = nn.GRU(input_size=inc,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  bias=bias,
                                  batch_first=batch_first,
                                  dropout=dropout,
                                  bidirectional=bidirectional)
        if self.task == 'va':
            self.v_head = Regressor(hidden_size * (bidirectional + 1), out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=True)
            self.a_head = Regressor(hidden_size * (bidirectional + 1), out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=True)
        else:
            self.fc = Regressor(hidden_size * (bidirectional + 1), out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)

    def forward(self, x, hp=None):

        self.gru_encoder.flatten_parameters()

        if self.affine_dim:
            x = self.affine(x)

        if self.task == 'va':
            output, hn = self.gru_encoder(x, hp)
            v_out = self.v_head(output)
            a_out = self.a_head(output)
            out = torch.cat([v_out, a_out], dim=-1)
            # return out, hn
            return out

        else:
            output, hn = self.gru_encoder(x, hp)
            out = self.fc(output)

            # return out, hn
            return out
