#!/bin/bash
#SBATCH --job-name=tiny-imagenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Carga solo si es necesario
# module load python/3.9.12  # Usar si no tienes python3.9 por defecto

VENV_DIR="$HOME/tiny_imagenet_venv"

# Activar entorno
source "$VENV_DIR/bin/activate"

# Ir a la carpeta con el script
cd "$HOME/lab9"

# Ejecutar script
python image_train3.py

