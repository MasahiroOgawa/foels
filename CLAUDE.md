# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **foels** project, a 4D computer vision reconstruction system implementing the FOELS (Focus of Expansion based Localized Segmentation) pipeline for detecting and tracking moving objects in video sequences.

## Development Setup

### Environment Setup
```bash
# Primary (UV - modern Python package management)
uv sync
source .venv/bin/activate

# Legacy environments (if working with specific models)
conda activate InternImage  # For InternImage model
conda activate unsupervised_detection  # For flow models
```

### Essential Commands

#### Running the Pipeline
```bash
# Main FOELS pipeline execution
./script/run_foels.sh [input_video_or_images] [result_directory]

# Dataset evaluation
./script/reconstruct/davis.sh        # DAVIS2016 dataset
./script/reconstruct/fbms.sh         # FBMS dataset
./script/reconstruct/process_SegTrackV2.sh  # SegTrackV2 dataset
```

#### Development Commands
```bash
# Linting (Ruff is installed via UV)
ruff check .
ruff format .

# Run specific module tests
python -m foels.optical_flow.flow_calc [input] [output]
python -m foels.camera_motion.calculate_camera_mat [args]

# Debug mode with detailed logging
LOG_LEVEL=5 ./script/run_foels.sh [input] [output]
```

## Architecture

### Pipeline Components (Sequential Processing)
1. **Optical Flow** (`foels/optical_flow/`) - Calculates pixel motion using UniMatch or RAFT
2. **Camera Motion** (`foels/camera_motion/`) - Estimates global camera movement
3. **FoE Calculation** (`foels/foe/`) - Computes Focus of Expansion points
4. **Moving Pixel Detection** (`foels/movingpixel_detection/`) - Identifies moving vs static pixels
5. **Object Refinement** (`foels/object_refinement/`) - Segments and refines moving objects using InternImage/OneFormer

### Key Configuration
- **YAML Parameters**: `script/params_*.yml` - Controls pipeline behavior
- **Model Selection**: Configure segmentation backend (internimage, oneformer) in YAML
- **Logging Levels**: 1 (basic) to 5 (interactive debug with matplotlib)

### External Dependencies (Git Submodules)
- `InternImage/` - Image segmentation model
- `UniMatch/` - Optical flow estimation
- `unsupervised_detection/` - Additional detection models

### Important Files
- `foels/FOELS.py` - Main pipeline orchestrator
- `script/config.py` - Configuration management
- `result/*/result.mp4` - Final output videos with overlays

## Development Notes

### Working with Models
- Models auto-download from Hugging Face on first run
- Check `~/flow_model/` and `~/.cache/huggingface/` for cached models
- GPU recommended but CPU fallback available

### Output Structure
Results are saved to specified directory with:
- `result.mp4` - Final video with moving object overlays
- `flow_*.pkl` - Cached optical flow data
- `mask_*.png` - Frame-by-frame segmentation masks
- Intermediate debug outputs when LOG_LEVEL > 3

### Common Debugging
- Use `LOG_LEVEL=5` for interactive matplotlib debugging
- Check `foels/util/logger.py` for logging configuration
- VS Code launch configs available in `.vscode/launch.json`