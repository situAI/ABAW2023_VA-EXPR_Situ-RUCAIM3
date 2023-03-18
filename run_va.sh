CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --master_port 9998 --nproc_per_node=2 main.py --config ./config/va.yaml >> va_history.out &
