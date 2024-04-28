#!/bin/bash 
#SBATCH -J mol_instruction1
#SBATCH -p gpu1
#SBATCH -N 1
# #SBATCH -w node[83]
#SBATCH --mem 60G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=icoxia@gmail.com
#SBATCH --output=log.%j.out 
#SBATCH --error=log.%j.err 
#SBATCH --cpus-per-task=8
module load anaconda3/2021.05 
source activate LAVIS

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/mol_instruction_stage1.yaml
