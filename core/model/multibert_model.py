from .bert_model import get_sinusoid_encoding_table, TransformerEncoder, Regressor
from utils.registery import MODEL_REGISTRY
import torch
import torch.nn as nn

class CrossFormerModule(nn.Module):
    """Cross Former Module

    Attributes:
        attn: multi-head attention module
        linear1: FC layer
        dropout: dropout layer
        linear2: FC layer
        norm1: LN layer
        norm2: LN layer
        dropout1: dropout layer
        dropout2: dropout layer
        activation: activation function
    """

    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        """Init module

        Args:
            d_model (int): input dim
            nheads (int): head num
            dim_feedforward (int): feedforward dim
            dropout (float): dropout ratio
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=nheads,
                                          dropout=dropout,
                                          bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def _sa_block(self, q, k, v):
        """Self-Attention Moudle (Transformer Encoder First Module)

        Args:
            q (torch.Tensor): query matrix
            k (torch.Tensor): key matrix
            v (torch.Tensor): value matrix

        Returns:
            out (torch.Tensor)
        """
        x = self.attn(query=q, key=k, value=v, need_weights=False)[0]

        return self.dropout1(x)

    def _ff_block(self, x):
        """FeedForward Module (Transformer Encoder Second Module)

        Args:
            x (torch.Tensor): input tensor

        Returns:
            out (torch.Tensor)
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout2(x)

    def forward(self, src, q, k, v):
        x = src
        x = self.norm1(x + self._sa_block(q, k, v))
        x = self.norm2(x + self._ff_block(x))

        return x

class CrossFormer(nn.Module):
    """CrossFormer V1

    Attributes:
        nlayers: num layers
        visual_branch: visual block
        audio_branch: audio block
    """

    def __init__(self, inc, nheads, feedforward_dim, nlayers, dropout):
        """Init CrossFormer V1

        Args:
            inc (int): input dim
            nheads (int): num heads
            feedforward_dim (int): feedforward dim
            nlayers (int): CrossFormer layer
            dropout (float): dropout ratio
        """
        super().__init__()
        self.nlayers = nlayers
        self.visual_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc, nheads=nheads, dim_feedforward=feedforward_dim, dropout=dropout) for _ in range(nlayers)
        ])

        self.audio_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc, nheads=nheads, dim_feedforward=feedforward_dim, dropout=dropout) for _ in range(nlayers)
        ])

    def forward(self, visual, audio):
        for i in range(self.nlayers):
            visual = self.visual_branch[i](src=visual, q=audio, k=visual, v=visual)
            audio = self.audio_branch[i](src=audio, q=visual, k=audio, v=audio)

        out = torch.cat([visual, audio], dim=-1)

        return out


class CrossFormerV2(nn.Module):
    """CrossFormer V2

    Attributes:
        nlayers: num layers
        visual_branch: visual block
        audio_branch: audio block
        visual_transformer: visual former module
        audio_transformer: audio former module
    """

    def __init__(self, inc, nheads, feedforward_dim, nlayers, dropout):
        """Init CrossFormerV2 Module

        Args:
            inc (int): input dim
            nheads (int): num heads
            feedforward_dim (int): feedforward dim
            nlayers (int): CrossFormer Module layers
            dropout (float): dropout ratio
        """
        super().__init__()
        self.nlayers = nlayers
        self.visual_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc, nheads=nheads, dim_feedforward=feedforward_dim, dropout=dropout) for _ in range(nlayers)
        ])

        self.audio_branch = nn.ModuleList([
            CrossFormerModule(d_model=inc, nheads=nheads, dim_feedforward=feedforward_dim, dropout=dropout) for _ in range(nlayers)
        ])

        self.visual_transformer = TransformerEncoder(inc, nheads, feedforward_dim, nlayers, dropout)
        self.audio_transformer = TransformerEncoder(inc, nheads, feedforward_dim, nlayers, dropout)

    def forward(self, visual, audio):
        for i in range(self.nlayers):
            visual = self.visual_branch[i](src=visual, q=audio, k=visual, v=visual)
            audio = self.audio_branch[i](src=audio, q=visual, k=audio, v=audio)

        visual = self.visual_transformer(visual)
        audio = self.audio_transformer(audio)

        out = torch.cat([visual, audio], dim=-1)

        return out


@MODEL_REGISTRY.register()
class MultiBERT(nn.Module):
    """MultiBERT V1 Module which share Self-Attention

    Attributes:
        input_dim: input dim
        feedforward_dim: feedforward_dim
        task: task str
        use_pe: wheather to use `position_embeddings`
        crossformer: crossformer module
        transformer: transformer module
    """

    def __init__(self,
                 input_dim,
                 feedforward_dim,
                 nheads,
                 nlayers,
                 dropout,
                 use_pe,
                 seq_len,
                 head_dropout,
                 head_dims,
                 out_dim,
                 task):
        """Init MultiBERT V1 Module

        Args:
            input_dim (int): input dim
            feedforward_dim (int): feedforward dim
            nheads (int): num heads
            nlayers (int): total layers
            dropout (float): encoder dropout ratio
            use_pe (bool): wheather to use `position_embeddings`
            seq_len (int): sequence length
            head_dropout (float): head block dropout ratio
            head_dims (int | list(int)): head dim list like. [512, 256]
            out_dim (int): a branch output dim
            task (str): task str
        """
        super().__init__()

        self.input_dim = input_dim
        inc = input_dim
        self.feedforward_dim = feedforward_dim
        self.task = task

        self.use_pe = use_pe
        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc // 2), freeze=True)

        self.crossformer = CrossFormer(inc=inc // 2, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers // 2, dropout=dropout)
        self.transformer = TransformerEncoder(inc=inc, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers // 2, dropout=dropout)

        if self.task == 'va':
            self.v_head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.a_head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
        else:
            self.head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)

    def forward(self, x):
        seq_len, bs, _ = x.shape

        multi_feature = x.split(x.shape[-1] // 2, -1)
        visual_feature = multi_feature[0]
        audio_feature = multi_feature[1]

        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            visual_feature = visual_feature + position_embeddings
            audio_feature = audio_feature + position_embeddings

        out = self.crossformer(visual=visual_feature, audio=audio_feature)
        out = self.transformer(out)

        if self.task == 'va':
            v_out = self.v_head(out)
            a_out = self.a_head(out)
            out = torch.cat([v_out, a_out], dim=-1)
        else:
            out = self.head(out)

        return out

@MODEL_REGISTRY.register()
class MultiBERTV2(nn.Module):
    """MultiBERT V2 Module which didn't share self-attention

    Attributes:
        input_dim: input dim
        feedforward_dim: feedforward dim
        task: task str
        use_pe: wheather to use `position_embeddings`
        crossformer: crossformer module
    """

    def __init__(self,
                 input_dim,
                 feedforward_dim,
                 nheads,
                 nlayers,
                 dropout,
                 use_pe,
                 seq_len,
                 head_dropout,
                 head_dims,
                 out_dim,
                 task):
        """Init MultiBERT V2 Module

        Args:
            input_dim (int): input dim
            feedforward_dim (int): feedforward dim
            nheads (int): num heads
            nlayers (int): total layers
            dropout (float): encoder dropout ratio
            use_pe (bool): wheather to use `position_embeddings`
            seq_len (int): sequence length
            head_dropout (float): head block dropout ratio
            head_dims (int | list(int)): head dim list like. [512, 256]
            out_dim (int): a branch output dim
            task (str): task str
        """
        super().__init__()

        self.input_dim = input_dim
        inc = input_dim
        self.feedforward_dim = feedforward_dim
        self.task = task

        self.use_pe = use_pe
        if use_pe:
            self.pe = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(seq_len, inc // 2), freeze=True)

        self.crossformer = CrossFormerV2(inc=inc // 2, nheads=nheads, feedforward_dim=feedforward_dim, nlayers=nlayers // 2, dropout=dropout)

        if self.task == 'va':
            self.v_head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
            self.a_head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU())
        else:
            self.head = Regressor(inc_dim=inc, out_dim=out_dim, dims_list=head_dims, dropout=head_dropout, act=nn.ReLU(), has_tanh=False)

    def forward(self, x):
        seq_len, bs, _ = x.shape

        multi_feature = x.split(x.shape[-1] // 2, -1)
        visual_feature = multi_feature[0]
        audio_feature = multi_feature[1]

        if self.use_pe:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(1).expand([seq_len, bs])
            position_embeddings = self.pe(position_ids)
            visual_feature = visual_feature + position_embeddings
            audio_feature = audio_feature + position_embeddings

        out = self.crossformer(visual=visual_feature, audio=audio_feature)

        if self.task == 'va':
            v_out = self.v_head(out)
            a_out = self.a_head(out)
            out = torch.cat([v_out, a_out], dim=-1)
        else:
            out = self.head(out)

        return out
