#!/bin/bash
set -e

echo "Setting up foels environment..."

# Check if we're in the right directory
if [ -f "/workspace/pyproject.toml" ]; then
    cd /workspace
    
    # Initialize git submodules
    if [ -d ".git" ]; then
        echo "Initializing git submodules..."
        git submodule update --init --recursive || true
    fi
    
    # Run uv sync to set up the Python environment
    echo "Running uv sync to install Python dependencies..."
    uv sync
    
    # Activate the virtual environment
    source .venv/bin/activate
    
    # Pre-download model weights from Hugging Face before firewall is activated
    echo "Pre-downloading model weights from Hugging Face..."
    python3 -c "
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
try:
    print('Downloading OneFormer model...')
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_coco_swin_large')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_coco_swin_large')
    print('OneFormer model downloaded successfully')
except Exception as e:
    print(f'Warning: Could not download OneFormer model: {e}')

try:
    print('Downloading UniMatch model...')
    import torch
    # The UniMatch model is usually downloaded when first used
    # We'll try to trigger the download here if possible
    model_path = os.path.expanduser('~/flow_model/')
    os.makedirs(model_path, exist_ok=True)
    print('UniMatch model directory prepared')
except Exception as e:
    print(f'Warning: Could not prepare UniMatch model: {e}')
" || echo "Warning: Some models could not be pre-downloaded. They will be downloaded on first use."
    
    echo "foels environment setup complete!"
    echo "To activate the environment, run: source .venv/bin/activate"
else
    echo "Note: foels project files not found. Run 'uv sync' manually after cloning the repository."
fi