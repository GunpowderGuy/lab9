#!/bin/bash

VENV_DIR="$HOME/tiny_imagenet_venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3.9 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install torch torchvision matplotlib
else
  echo "Virtual environment already exists. Skipping creation."
fi
