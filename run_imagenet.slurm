#!/bin/bash
#SBATCH --job-name=tiny-imagenet
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# load module if needed
module load python/3.10    # adjust to your cluster's module

# define and create venv if missing
VENV_DIR=$HOME/tiny_imagenet_venv
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install torch torchvision pillow lion-pytorch
else
  source "$VENV_DIR/bin/activate"
fi

# run your training script
srun python imagenet_train.py
