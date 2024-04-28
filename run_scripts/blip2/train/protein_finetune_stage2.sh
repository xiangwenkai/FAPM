#!/bin/bash
#SBATCH -J ft
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -w node[83]
#SBATCH --mem 200G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=icoxia@gmail.com
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --cpus-per-task=8
module load anaconda3/2021.05
source activate LAVIS

python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/finetune_stage2.yaml
