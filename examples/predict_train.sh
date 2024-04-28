#!/bin/bash
#SBATCH -J infer_cc
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -w node[84]
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH --output=log_predict.out
#SBATCH --error=log_predict.err
#SBATCH --cpus-per-task=8
module load anaconda3/2021.05
source activate LAVIS

python blip2_predict_func_concat.py
