import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.registery import DATASET_REGISTRY, COLLATE_FN_REGISTRY
from .collate_fn import base_collate_fn, image_collate_fn

from .SequenceData import SequenceData


def build_dataset(cfg, prefix):

    dataset_cfg = copy.deepcopy(cfg)
    try:
        dataset_cfg = dataset_cfg[prefix]
    except Exception:
        raise f'should contain {prefix}!'

    data = DATASET_REGISTRY.get(dataset_cfg['name'])(**dataset_cfg['args'])

    return data


def build_dataloader(cfg):

    if torch.distributed.is_initialized():
        ddp = True
    else:
        ddp = False

    test_flag = False
    if 'test_data' in cfg.keys():
        test_flag = True
    else:
        test_flag = False

    dataloader_cfg = copy.deepcopy(cfg)
    try:
        dataloader_cfg = cfg['dataloader']
    except Exception:
        raise 'should contain {dataloader}!'

    if ddp:
        if test_flag == False:
            train_ds = build_dataset(cfg, 'train_data')
            val_ds = build_dataset(cfg, 'val_data')

            train_sampler = DistributedSampler(train_ds)
            collate_fn = COLLATE_FN_REGISTRY.get(dataloader_cfg.pop('collate_fn'))

            train_loader = DataLoader(train_ds,
                                      sampler=train_sampler,
                                      collate_fn=collate_fn,
                                      **dataloader_cfg)

            val_loader = DataLoader(val_ds,
                                    collate_fn=collate_fn,
                                    **dataloader_cfg)

            return train_loader, val_loader
        else:
            test_ds = build_dataset(cfg, 'test_data')
            collate_fn = COLLATE_FN_REGISTRY.get(dataloader_cfg.pop('collate_fn'))
            test_loader = DataLoader(test_ds,
                                     collate_fn=collate_fn,
                                     **dataloader_cfg)

            return None, test_loader
    else:
        if test_flag == False:
            val_ds = build_dataset(cfg, 'val_data')

            collate_fn = COLLATE_FN_REGISTRY.get(dataloader_cfg.pop('collate_fn'))

            val_loader = DataLoader(val_ds,
                                    collate_fn=collate_fn,
                                    **dataloader_cfg)

            return None, val_loader
        else:
            test_ds = build_dataset(cfg, 'test_data')
            collate_fn = COLLATE_FN_REGISTRY.get(dataloader_cfg.pop('collate_fn'))
            test_loader = DataLoader(test_ds,
                                     collate_fn=collate_fn,
                                     **dataloader_cfg)

            return None, test_loader
