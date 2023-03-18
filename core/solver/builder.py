from utils.registery import SOLVER_REGISTRY
import copy

from .base_solver import BaseSolver
from .rdrop_solver import RDropSolver

def build_solver(cfg):
    cfg = copy.deepcopy(cfg)

    try:
        solver_cfg = cfg['solver']
    except Exception:
        raise 'should contain {solver}!'

    return SOLVER_REGISTRY.get(solver_cfg['name'])(cfg)
