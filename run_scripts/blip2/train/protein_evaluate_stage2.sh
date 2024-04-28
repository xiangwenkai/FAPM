#!/bin/bash
#SBATCH -J case
#SBATCH -p gpu1
#SBATCH -N 1
# #SBATCH -w node[85]
#SBATCH --mem 50G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=icoxia@gmail.com
#SBATCH --output=log_eval.out
#SBATCH --error=log_eval.err
#SBATCH --cpus-per-task=8
module load anaconda3/2021.05
source activate LAVIS

python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/test_stage2_evaluate.yaml
