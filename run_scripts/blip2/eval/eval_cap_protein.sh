#!/bin/bash
#SBATCH -J get_eval1
#SBATCH -p nvidia
#SBATCH -N 1
#SBATCH -w node[87]
#SBATCH --mem 40G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=icoxia@gmail.com
#SBATCH --output=log_eval1.out
#SBATCH --error=log_eval1.err
#SBATCH --cpus-per-task=4
module load anaconda3/2021.05
source activate LAVIS

python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/caption_protein_eval.yaml
