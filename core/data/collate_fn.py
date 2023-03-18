import torch
from utils.registery import COLLATE_FN_REGISTRY
import numpy as np


@COLLATE_FN_REGISTRY.register()
def base_collate_fn(batch):
    feats, labels, seq_list = list(), list(), list()
    for crt_feat, crt_label, crt_seq in batch:
        feats.append(crt_feat)
        labels.append(crt_label)
        seq_list.append(crt_seq)

    feats = torch.from_numpy(np.asarray(feats)).transpose(0, 1)
    labels = torch.from_numpy(np.asarray(labels)).transpose(0, 1)
    seq_list = np.asarray(seq_list).transpose()

    return {'feat': feats, 'label': labels, 'seq_list': seq_list}

@COLLATE_FN_REGISTRY.register()
def test_collate_fn(batch):
    feats, seq_list = list(), list()
    for crt_feat, crt_seq in batch:
        feats.append(crt_feat)
        seq_list.append(crt_seq)

    feats = torch.from_numpy(np.asarray(feats)).transpose(0, 1)
    seq_list = np.asarray(seq_list).transpose()

    return {'feat': feats, 'seq_list': seq_list}
