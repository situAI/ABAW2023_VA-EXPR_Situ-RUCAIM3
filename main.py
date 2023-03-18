import yaml
from core.solver import build_solver
import torch
import numpy as np
import random
import argparse


def init_seed(seed=778):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/baseline.yaml', type=str, help='config file')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--ensemble', action='store_true', default=False, help='wheather to run ensemble')
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r').read(), Loader=yaml.FullLoader)
    init_seed(cfg['seed'])

    if not args.ensemble:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

        solver = build_solver(cfg)

        solver.run()
    else:
        solver = build_solver(cfg)
        solver.run()


if __name__ == '__main__':
    main()
