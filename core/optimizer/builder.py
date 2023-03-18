import torch
import timm
import timm.scheduler
import inspect
from utils.registery import OPTIMIZER_REGISTRY, LR_SCHEDULER_REGISTRY
import copy

def register_torch_optimizers():
    """
    Register all optimizers implemented by torch
    """
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZER_REGISTRY.register()(_optim)

def build_optimizer(cfg):
    register_torch_optimizers()
    optimizer_cfg = copy.deepcopy(cfg)

    try:
        optimizer_cfg = cfg['solver']['optimizer']
    except Exception:
        raise 'should contain {solver.optimizer}!'

    return OPTIMIZER_REGISTRY.get(optimizer_cfg['name'])

def register_torch_lr_scheduler():
    """
    Register all lr_schedulers implemented by torch
    """
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith('__'):
            continue

        _scheduler = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_scheduler) and issubclass(_scheduler, torch.optim.lr_scheduler._LRScheduler):
            LR_SCHEDULER_REGISTRY.register()(_scheduler)

def register_timm_lr_scheduler():
    """
    Register all lr_schedulers implemented by timm
    """
    for module_name in dir(timm.scheduler):
        if module_name.startswith('__') or 'create' in module_name:
            continue

        _scheduler = getattr(timm.scheduler, module_name)
        if inspect.isclass(_scheduler) and issubclass(_scheduler, timm.scheduler.scheduler.Scheduler):
            LR_SCHEDULER_REGISTRY.register()(_scheduler)

def build_lr_scheduler(cfg):
    register_torch_lr_scheduler()
    register_timm_lr_scheduler()
    scheduler_cfg = copy.deepcopy(cfg)

    try:
        scheduler_cfg = cfg['solver']['lr_scheduler']
    except Exception:
        raise 'should contain {solver.lr_scheduler}!'

    return LR_SCHEDULER_REGISTRY.get(scheduler_cfg['name'])
