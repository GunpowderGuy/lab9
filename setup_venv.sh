#!/bin/bash

VENV_DIR="$HOME/tiny_imagenet_venv"

# Crear el entorno virtual si no existe
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3.9 -m venv "$VENV_DIR"
fi

# Activar entorno
source "$VENV_DIR/bin/activate"

# Instalar/actualizar dependencias siempre
pip install --upgrade pip
pip install torch torchvision matplotlib lion-pytorch
