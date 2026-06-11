# MemFlow Integration Summary

## Overview
MemFlow has been successfully integrated into the foels project as an alternative optical flow method to Unimatch.

## Installation

### 1. MemFlow is added as a git submodule
Located at: `foels/ext/memflow`

### 2. Environment Setup
MemFlow requires a conda environment (uv was attempted but incompatible with CUDA packages):
```bash
conda create --name memflow python=3.8
conda activate memflow
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install yacs loguru einops timm==0.4.12 imageio matplotlib tensorboard scipy opencv-python h5py tqdm
```

### 3. Model Weights
Pre-trained model downloaded to: `foels/ext/memflow/ckpts/MemFlowNet_things.pth`

## Configuration

In `script/foels_param.yaml`, you can select the optical flow method:

```yaml
OpticalFlow:
  flow_type: "memflow"  # or "unimatch"

  # MemFlow settings
  memflow_model: "MemFlowNet"
  memflow_stage: "things"
  memflow_weights: "foels/ext/memflow/ckpts/MemFlowNet_things.pth"

  # Unimatch settings (fallback)
  unimatch_model: "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
```

## GPU Requirements

**IMPORTANT**: MemFlow requires an NVIDIA GPU with CUDA support. The script will:
- Check for GPU availability when MemFlow is selected
- Display an error message if no GPU is found
- Suggest switching to Unimatch as an alternative

## Usage

1. To use MemFlow (requires GPU):
   - Set `flow_type: "memflow"` in `script/foels_param.yaml`
   - Run: `./script/run_foels.sh`

2. To use Unimatch (CPU compatible):
   - Set `flow_type: "unimatch"` in `script/foels_param.yaml`
   - Run: `./script/run_foels.sh`

## Testing

A standalone test script is available:
```bash
/home/mas/anaconda3/envs/memflow/bin/python test_memflow.py
```

## Implementation Details

1. **opticalflow.py**: Added `MemFlow` class with methods:
   - `compute_from_images()`: Process image file pairs
   - `compute_from_arrays()`: Process numpy array pairs

2. **run_foels.sh**: Updated to:
   - Detect flow type from configuration
   - Check GPU availability for MemFlow
   - Handle environment switching between conda and venv
   - Provide clear error messages for GPU requirements

3. **inference_wrapper.py**: Created CPU/GPU compatible wrapper for MemFlow inference

## Notes

- MemFlow provides state-of-the-art optical flow estimation with memory-based approach
- Performance is significantly better on GPU compared to CPU
- The integration maintains backward compatibility with existing Unimatch workflow