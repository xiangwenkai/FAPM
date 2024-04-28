#!/bin/bash
#SBATCH -J secc2
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -w node[84]
#SBATCH --mem 80G
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=icoxia@gmail.com
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --cpus-per-task=8
module load anaconda3/2021.05
source activate LAVIS

python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/union_stage2.yaml
