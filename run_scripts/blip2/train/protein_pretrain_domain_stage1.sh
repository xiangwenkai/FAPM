#!/bin/bash 
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/protein_pretrain_stage1.yaml
