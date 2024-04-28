#!/bin/bash
#SBATCH -J finetune_function
#SBATCH -p Jupyter
#SBATCH -N 1
#SBATCH -w node[39]
#SBATCH --mem 260G
#SBATCH --gres=gpu:7
#SBATCH --mail-type=ALL
#SBATCH --mail-user=icoxia@gmail.com
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --cpus-per-task=8
module load anaconda3/2021.05
source activate LAVIS

python -m torch.distributed.run --nproc_per_node=7 train.py --cfg-path lavis/projects/blip2/train/reviewed_function_ft.yaml
