import torch
import inspect
from utils.registery import LOSS_REGISTRY
import copy

from .loss import CCCLoss, ExprLoss

def register_torch_loss():
    for module_name in dir(torch.nn):
        if module_name.startswith('__') or 'Loss' not in module_name:
            continue
        _loss = getattr(torch.nn, module_name)
        if inspect.isclass(_loss) and issubclass(_loss, torch.nn.Module):
            LOSS_REGISTRY.register()(_loss)

def build_loss(cfg):
    register_torch_loss()
    loss_cfg = copy.deepcopy(cfg)

    try:
        loss_cfg = cfg['solver']['loss']
    except Exception:
        raise 'should contain {solver.loss}!'
    
    return LOSS_REGISTRY.get(loss_cfg['name'])(**loss_cfg['args'])
