CUDA_VISIBLE_DEVICES=6,7 nohup python -m torch.distributed.launch --master_port 9997 --nproc_per_node=2 main.py --config ./config/expr_rdrop.yaml >> expr_history.out &
