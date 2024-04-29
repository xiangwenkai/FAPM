#!/bin/bash
python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/union_stage2.yaml
