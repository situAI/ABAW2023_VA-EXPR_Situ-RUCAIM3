from utils.registery import METRIC_REGISTRY
import copy

from .metrics import AUMetric, VAMetric, ExprMetric

def build_metric(cfg):
    cfg = copy.deepcopy(cfg)
    try:
        metric_cfg = cfg['solver']['metric']
    except Exception:
        raise 'should contain {solver.metric}!'

    return METRIC_REGISTRY.get(metric_cfg['name'])()
