#!/bin/bash
###
USAGE="Usage: $0 [input image directory or movie (default: from foels_param.yaml)] [result directory (default: from foels_param.yaml)]"
echo $USAGE

# -e: stop immediately when error occurred
# -u: stop immediately when undefined variable is used
set -eu

# set root directory (get absolute path)
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)



# Load parameters from YAML file
PARAM_FILE="${ROOT_DIR}/script/foels_param.yaml"

# Read input_dir from YAML file if no argument provided
if [ $# -eq 0 ]; then
    INPUT_FROM_YAML=$(yq -r '.MovingObjectExtractor.input_dir' "${PARAM_FILE}")
    # Convert relative path to absolute path if needed
    if [[ ! "$INPUT_FROM_YAML" = /* ]]; then
        INPUT="${ROOT_DIR}/${INPUT_FROM_YAML}"
    else
        INPUT="${INPUT_FROM_YAML}"
    fi
else
    INPUT="${1}"
fi

# Read result_dir from YAML file if no second argument provided
if [ $# -le 1 ]; then
    RESULT_FROM_YAML=$(yq -r '.MovingObjectExtractor.result_dir' "${PARAM_FILE}")
    # Convert relative path to absolute path if needed
    if [[ ! "$RESULT_FROM_YAML" = /* ]]; then
        RESULT_PARENT_DIR="${ROOT_DIR}/${RESULT_FROM_YAML}"
    else
        RESULT_PARENT_DIR="${RESULT_FROM_YAML}"
    fi
else
    RESULT_PARENT_DIR="${2}"
fi

# Read parameters from YAML
LOG_LEVEL=$(yq -r ' .MovingObjectExtractor.loglevel' "${PARAM_FILE}")
IMG_WIDTH=$(yq -r ' .CreateImage.width ' "$PARAM_FILE")
IN_FPS=$(yq -r ' .CreateImage.fps ' "$PARAM_FILE")
SKIP_FRAMES=$(yq -r ' .MovingObjectExtractor.skip_frames ' "$PARAM_FILE")
SEG_MODEL_NAME=$(yq -r ' .MovingObjectExtractor.segment_model_name ' "$PARAM_FILE")
# Strip quotes and whitespace from SEG_MODEL_NAME
SEG_MODEL_NAME=$(echo "$SEG_MODEL_NAME" | sed -e 's/^\s*//;s/\s*$//' -e 's/^"//;s/"$//' -e "s/^'//;s/'$//")
NUM_RANSAC=$(yq ' .NUM_RANSAC ' "$PARAM_FILE")
# Read flow settings from YAML
FLOW_TYPE=$(yq -r ' .OpticalFlow.flow_type ' "$PARAM_FILE")
# Strip quotes and whitespace from FLOW_TYPE
FLOW_TYPE=$(echo "$FLOW_TYPE" | sed -e 's/^\s*//;s/\s*$//' -e 's/^"//;s/"$//' -e "s/^'//;s/'$//")

# Read model-specific settings based on flow type
if [ "${FLOW_TYPE}" = "memflow" ]; then
    MEMFLOW_MODEL=$(yq -r ' .OpticalFlow.memflow_model ' "$PARAM_FILE")
    MEMFLOW_STAGE=$(yq -r ' .OpticalFlow.memflow_stage ' "$PARAM_FILE")
    MEMFLOW_WEIGHTS=$(yq -r ' .OpticalFlow.memflow_weights ' "$PARAM_FILE")
else
    # Default to unimatch
    FLOW_MODEL_NAME=$(yq -r ' .OpticalFlow.unimatch_model ' "$PARAM_FILE")
    # Strip quotes and whitespace from FLOW_MODEL_NAME
    FLOW_MODEL_NAME=$(echo "$FLOW_MODEL_NAME" | sed -e 's/^\s*//;s/\s*$//' -e 's/^"//;s/"$//' -e "s/^'//;s/'$//")
fi

RANSAC_ALL_INLIER_ESTIMATION=$(yq ' .RANSAC_ALL_INLIER_ESTIMATION ' "$PARAM_FILE")
FOE_SEARCH_STEP=$(yq ' .FOE_SEARCH_STEP ' "$PARAM_FILE")
THRE_MOVING_FRACTION_IN_OBJ=$(yq ' .THRE_MOVING_FRACTION_IN_OBJ ' "$PARAM_FILE")


if [ $LOG_LEVEL -ge 3 ]; then
       echo "[INFO] set debug mode"
       set -x
fi

# to define INPUT_DIR, we need to do below first.
echo "[INFO] check input is whether a directory or movie."
if [ -d "${INPUT}" ]; then
       echo "[INFO] input is a directory."
       INPUT_DIR="${INPUT}"

       # Validate that the directory contains image files
       image_count=$(find "${INPUT_DIR}" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)

       if [ "$image_count" -eq 0 ]; then
              # Check if there are video files instead
              video_files=$(find "${INPUT_DIR}" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) -printf "%f\n" 2>/dev/null | head -3)

              if [ -n "$video_files" ]; then
                     echo "[ERROR] The input directory contains video file(s) but no image sequences."
                     echo "[ERROR] Found video files: $(echo "$video_files" | paste -sd ', ')"
                     echo ""
                     echo "[INFO] Please provide either:"
                     echo "       1. A video file directly as input (not a directory containing videos)"
                     echo "          Example: ./script/run_foels.sh ~/data/video.mp4"
                     echo "       2. A directory containing image sequences (e.g., 00001.jpg, 00002.jpg, ...)"
                     echo "          Example: ./script/run_foels.sh ~/data/image_sequences/"
              else
                     echo "[ERROR] No image files found in directory: ${INPUT_DIR}"
                     echo "[ERROR] The directory must contain image files (.jpg, .jpeg, or .png)"
                     echo ""
                     echo "[INFO] If you have a video file, please provide it directly as input:"
                     echo "       Example: ./script/run_foels.sh ~/data/video.mp4"
              fi
              exit 1
       fi

       echo "[INFO] Found ${image_count} image file(s) in the directory."
elif [ -f "${INPUT}" ]; then
       echo "[INFO] input is a movie."
       echo "[INFO] convert movie to images"
       INPUT_DIR=$(dirname "${INPUT}")
       ffmpeg -i "${INPUT}" -r "${IN_FPS}" -vf scale="${IMG_WIDTH}":-1 "${INPUT_DIR}"/%06d.png
else
       echo "[ERROR] input is neither a directory nor a movie."
       exit 1
fi

# automatically defined variables from INPUT
RESULT_PARENT_DIR="${RESULT_PARENT_DIR}/$(basename "${INPUT_DIR}")"
RESULT_FLOW_DIR="${RESULT_PARENT_DIR}/flow"
RESULT_SEG_DIR="${RESULT_PARENT_DIR}/segmentation"
RESULT_MOVOBJ_DIR="${RESULT_PARENT_DIR}/moving_object"
case ${SEG_MODEL_NAME} in
       "upernet_internimage_t_512_160k_ade20k.pth" |\
       "upernet_internimage_xl_640_160k_ade20k.pth" |\
       "upernet_internimage_h_896_160k_ade20k.pth")
              SEG_MODEL_TYPE="internimage"
              SEG_CHECKPOINT_DIR="${ROOT_DIR}/foels/ext/InternImage/checkpoint_dir/seg"
              SEG_TASK_TYPE="semantic";;
       "mask_rcnn_internimage_t_fpn_1x_coco.pth")
              SEG_MODEL_TYPE="internimage"
              SEG_CHECKPOINT_DIR="${ROOT_DIR}/foels/ext/InternImage/checkpoint_dir/det"
              SEG_TASK_TYPE="instance";;
       "shi-labs/oneformer_coco_swin_large")
              SEG_MODEL_TYPE="oneformer"
              SEG_TASK_TYPE="panoptic";;
       *)
              echo "[ERROR] unknown segmentation model name: ${SEG_MODEL_NAME}"
              exit 1;;
esac


deactivate_all_environments() {
       local environment_deactivated=false

       # Check and deactivate virtual environment
       if [ -n "${VIRTUAL_ENV:-}" ]; then
              echo "[INFO] Deactivating venv: $VIRTUAL_ENV"
              if deactivate 2>/dev/null; then
                     environment_deactivated=true
              else
                     echo "[WARNING] Failed to deactivate venv"
              fi
       fi

       # Check and deactivate conda environment
       if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
              echo "[INFO] Deactivating conda env: $CONDA_DEFAULT_ENV"
              if conda deactivate 2>/dev/null; then
                     environment_deactivated=true
              else
                     echo "[WARNING] Failed to deactivate conda env"
              fi
       fi

       if [ "$environment_deactivated" = true ]; then
              echo "[INFO] Successfully deactivated environments"
       fi

       # Clean PATH
       export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "\.venv" | tr '\n' ':' | sed 's/:$//')
}


echo "[INFO] compute optical flow using ${FLOW_TYPE}"
# CMD_PREFIX can be used to set CUDA device if needed
# Example: CMD_PREFIX="env CUDA_VISIBLE_DEVICES=0"
CMD_PREFIX=""
if [ -d "${RESULT_FLOW_DIR}" ] && [ -n "$(ls -A "${RESULT_FLOW_DIR}"/*.mp4 2>/dev/null)" ]; then
       echo "[INFO] optical flow output files already exist. Skip computing optical flow."
else
       mkdir -p "${RESULT_FLOW_DIR}"
       export OMP_NUM_THREADS=1
       # to avoid CUDA out of memory error.
       export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

       if [ "${FLOW_TYPE}" = "memflow" ]; then
              echo "[INFO] using MemFlow for optical flow computation"

              # Check if GPU is available
              if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
                     echo "[ERROR] No NVIDIA GPU detected. MemFlow requires a GPU for processing."
                     echo "[ERROR] Cannot proceed with MemFlow without GPU support."
                     echo "[INFO] Please either:"
                     echo "       1. Run this on a system with an NVIDIA GPU, or"
                     echo "       2. Change 'flow_type' to 'unimatch' in script/foels_param.yaml"
                     exit 1
              else
                     # GPU is available, proceed with MemFlow
                     # Activate memflow conda environment
                     set +eu
                     deactivate_all_environments
                     source "$(conda info --base)/etc/profile.d/conda.sh"
                     conda activate memflow
                     set -eu

                     # Run MemFlow inference (need to cd to memflow directory for imports to work)
                     cd "${ROOT_DIR}/foels/ext/memflow"
                     # Set memory optimization for CUDA (PyTorch 1.13 compatible)
                     export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
                     export CUDA_LAUNCH_BLOCKING=0
                     # Use absolute path for input directory (not the original INPUT which might be a video file)
                     if [[ "${INPUT_DIR}" != /* ]]; then
                            INPUT_ABS="${ROOT_DIR}/${INPUT_DIR}"
                     else
                            INPUT_ABS="${INPUT_DIR}"
                     fi
                     # Use absolute path for weights
                     if [[ "${MEMFLOW_WEIGHTS}" != /* ]]; then
                            WEIGHTS_ABS="${ROOT_DIR}/${MEMFLOW_WEIGHTS}"
                     else
                            WEIGHTS_ABS="${MEMFLOW_WEIGHTS}"
                     fi
                     # Use absolute path for output directory
                     if [[ "${RESULT_FLOW_DIR}" != /* ]]; then
                            OUTPUT_ABS="${ROOT_DIR}/${RESULT_FLOW_DIR}"
                     else
                            OUTPUT_ABS="${RESULT_FLOW_DIR}"
                     fi
                     # Use optimized inference wrapper for better memory management
                     python inference_wrapper_optimized.py \
                     --name "${MEMFLOW_MODEL}" \
                     --stage "${MEMFLOW_STAGE}" \
                     --restore_ckpt "${WEIGHTS_ABS}" \
                     --seq_dir "${INPUT_ABS}" \
                     --vis_dir "${OUTPUT_ABS}"
                     cd "${ROOT_DIR}"

                     # Check if input images start from 00000 or 00001 to determine if renaming is needed
                     # MemFlow's inference_wrapper.py now automatically scans the input directory for image files,
                     # detects the starting index (e.g., 00000, 000000, or 00001), and generates flow files with the correct indexing.
                     if [ -f "${INPUT_ABS}/00000.jpg" ] || [ -f "${INPUT_ABS}/00000.png" ]; then
                         echo "[INFO] Input images start from 00000, MemFlow generated 0-indexed flow files correctly"
                     elif [ -f "${INPUT_ABS}/000000.jpg" ] || [ -f "${INPUT_ABS}/000000.png" ]; then
                         echo "[INFO] Input images start from 000000, MemFlow generated 0-indexed flow files correctly"
                     else
                         echo "[INFO] Input images start from 00001 or higher, MemFlow generated correctly indexed flow files"
                     fi

                     # Reactivate main environment
                     conda deactivate
                     source "${ROOT_DIR}/.venv/bin/activate"
              fi
       fi

       if [ "${FLOW_TYPE}" = "unimatch" ]; then
              echo "[INFO] using Unimatch for optical flow computation"
              source "${ROOT_DIR}/.venv/bin/activate"
              echo "[INFO] env: $VIRTUAL_ENV"

              if [ ! -f "${ROOT_DIR}/foels/ext/unimatch/pretrained/${FLOW_MODEL_NAME}" ]; then
                     echo "[INFO] download pretrained model"
                     mkdir -p "${ROOT_DIR}/foels/ext/unimatch/pretrained"
                     wget "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/${FLOW_MODEL_NAME}" -P "${ROOT_DIR}/foels/ext/unimatch/pretrained"
              fi

              python "${ROOT_DIR}/foels/ext/unimatch/main_flow.py" \
              --inference_dir "${INPUT_DIR}" \
              --output_path "${RESULT_FLOW_DIR}" \
              --resume "${ROOT_DIR}/foels/ext/unimatch/pretrained/${FLOW_MODEL_NAME}" \
              --inference_size 768 1024 \
              --padding_factor 32 \
              --upsample_factor 4 \
              --num_scales 2 \
              --attn_splits_list 2 8 \
              --corr_radius_list -1 4 \
              --prop_radius_list -1 1 \
              --reg_refine \
              --num_reg_refine 6 \
              --save_flo_flow
       fi

       echo "[INFO] save optical flow to ${RESULT_FLOW_DIR}"
       echo "[INFO] creating a flow movie"
       ffmpeg -framerate 30  -pattern_type glob -i "${RESULT_FLOW_DIR}/*.png" \
              -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${RESULT_FLOW_DIR}/flow.mp4
fi


echo "[INFO] run segmentation"
if [ -d ${RESULT_SEG_DIR} ] && [ -n "$(ls -A ${RESULT_SEG_DIR}/*.mp4)" ]; then
       echo "[INFO] segmentation output files already exist. Skip running segmentation."
else
       mkdir -p ${RESULT_SEG_DIR}
       case ${SEG_MODEL_TYPE} in
              "internimage")
                     echo "[INFO] activate InternImage conda env"
                     # to avoid error: "anaconda3/envs/internimage/etc/conda/activate.d/libblas_mkl_activate.sh: 
                     # line 1: MKL_INTERFACE_LAYER: unbound variable", we set +u.
                     set +eu
                     deactivate_allenvs
                     source $(conda info --base)/etc/profile.d/conda.sh
                     conda activate internimage
                     # eval "$(conda shell.bash activate internimage)"
                     set -eu
                     echo "[INFO] env: $CONDA_DEFAULT_ENV"
                     # swith python interpretor to the one in the conda env.       
                     PYTHON_INTERPRETER=$(which python)
                     echo "[INFO] python interpretor: $PYTHON_INTERPRETER"

                     if [ ! -f ${SEG_CHECKPOINT_DIR}/${SEG_MODEL_NAME} ]; then
                            echo "[INFO] download pretrained model"
                            mkdir -p ${SEG_CHECKPOINT_DIR}
                            # swith download link by the segmentation model
                            # add --content-disposition to prevent adding download=true in the downloded file name.
                            case ${SEG_MODEL_NAME} in
                                   "upernet_internimage_t_512_160k_ade20k.pth")
                                          wget --content-disposition https://huggingface.co/OpenGVLab/InternImage/resolve/fc1e4e7e01c3e7a39a3875bdebb6577a7256ff91/upernet_internimage_t_512_160k_ade20k.pth?download=true -P ${SEG_CHECKPOINT_DIR};;
                                   "upernet_internimage_xl_640_160k_ade20k.pth")
                                          wget --content-disposition https://huggingface.co/OpenGVLab/InternImage/resolve/fc1e4e7e01c3e7a39a3875bdebb6577a7256ff91/upernet_internimage_xl_640_160k_ade20k.pth -P ${SEG_CHECKPOINT_DIR};; 
                                   "upernet_internimage_h_896_160k_ade20k.pth")
                                          wget --content-disposition https://huggingface.co/OpenGVLab/InternImage/resolve/fc1e4e7e01c3e7a39a3875bdebb6577a7256ff91/upernet_internimage_h_896_160k_ade20k.pth?download=true -P ${SEG_CHECKPOINT_DIR};;
                                   "mask_rcnn_internimage_t_fpn_1x_coco.pth")
                                          wget --content-disposition https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.pth?download=true -P ${SEG_CHECKPOINT_DIR};;
                            *) echo "[ERROR] unknown segmentation model name: ${SEG_MODEL_NAME}"; exit 1;;
                            esac
                     fi

                     echo "[INFO] run segmentation using: ${SEG_MODEL_TYPE} ${SEG_TASK_TYPE}"
                     if [ "$SEG_TASK_TYPE" = "instance" ]; then
                            python "${ROOT_DIR}/foels/ext/InternImage/detection/image_demo.py" \
                            "${INPUT_DIR}" \
                            "${ROOT_DIR}/foels/ext/InternImage/detection/configs/coco/${SEG_MODEL_NAME%.*}.py" \
                            "${ROOT_DIR}/foels/ext/InternImage/checkpoint_dir/det/${SEG_MODEL_NAME}" \
                            --out "${RESULT_SEG_DIR}"
                     elif [ "$SEG_TASK_TYPE" = "semantic" ]; then
                            python "${ROOT_DIR}/foels/ext/InternImage/segmentation/image_demo.py" \
                                   "${INPUT_DIR}" \
                                   "${ROOT_DIR}/foels/ext/InternImage/segmentation/configs/ade20k/${SEG_MODEL_NAME%.*}.py" \
                                   "${ROOT_DIR}/foels/ext/InternImage/checkpoint_dir/seg/${SEG_MODEL_NAME}" \
                                   --palette ade20k --out "${RESULT_SEG_DIR}"
                     else
                            echo "[ERROR] unknown segmentation task type: ${SEG_TASK_TYPE}"
                            exit 1
                     fi
                     # Deactivate all environments using the provided function
                     deactivate_allenvs
                     ;;
              "oneformer")
                     echo "[INFO] you choose segmentation: ${SEG_MODEL_TYPE} ${SEG_TASK_TYPE}\
                      The process will be done during moving object extraction."

       esac
fi


echo "[INFO] run extract moving objects"
source "${ROOT_DIR}/.venv/bin/activate"
echo "[INFO] env: $VIRTUAL_ENV"

# Count input frames
NUM_INPUT_FRAMES=$(ls -1 "${INPUT_DIR}"/*.{jpg,png} 2>/dev/null | wc -l)

if [ "$SKIP_FRAMES" -ge "$NUM_INPUT_FRAMES" ]; then
    echo "[WARNING] SKIP_FRAMES (${SKIP_FRAMES}) >= number of input frames (${NUM_INPUT_FRAMES}). No frames will be processed. Skipping moving object extraction."
    exit 0
else
    mkdir -p "${RESULT_MOVOBJ_DIR}"
    if [ $LOG_LEVEL -ge 5 ]; then
       echo "[NOTE] Please press F5 to start debugging!"
       python -Xfrozen_modules=off -m debugpy --listen 5678 --wait-for-client "${ROOT_DIR}/foels/extract_moving_objects.py" \
       --config "${PARAM_FILE}" \
       --input_dir "${INPUT_DIR}" \
       --flow_result_dir "${RESULT_FLOW_DIR}" \
       --segment_result_dir "${RESULT_SEG_DIR}" \
       --result_dir "${RESULT_MOVOBJ_DIR}"
    else
       python "${ROOT_DIR}/foels/extract_moving_objects.py" \
       --config "${PARAM_FILE}" \
       --input_dir "${INPUT_DIR}" \
       --flow_result_dir "${RESULT_FLOW_DIR}" \
       --segment_result_dir "${RESULT_SEG_DIR}" \
       --result_dir "${RESULT_MOVOBJ_DIR}"
    fi
fi


echo "[INFO] creating a segmentation movie (ffmpeg in InternImage conda env doesn't support libx264, so we create it here.)"
OUT_FPS=$(yq -r '.CreateVideo.fps' "${PARAM_FILE}")
# for segmentation, the image file format is jpg or png. so detect it first.
IMG_EXT=
if [ $(ls -1 ${RESULT_SEG_DIR}/*.jpg 2>/dev/null | wc -l) != 0 ]; then
       IMG_EXT=jpg
elif [ $(ls -1 ${RESULT_SEG_DIR}/*.png 2>/dev/null | wc -l) != 0 ]; then
       IMG_EXT=png
else
       echo "[INFO] no jpg or png image file in ${RESULT_SEG_DIR}\
        So skip creating a segmentation movie."
fi
if [ $(ls -1 ${RESULT_SEG_DIR}/*.${IMG_EXT} 2>/dev/null | wc -l) != 0 ]; then
       ffmpeg -y -framerate ${OUT_FPS}  -pattern_type glob -i "${RESULT_SEG_DIR}/*.${IMG_EXT}" \
       -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${RESULT_SEG_DIR}/segmentation.mp4
fi

echo "[INFO] creating a final movie"
RESULT_FILENAME=$(yq -r '.CreateVideo.result_filename' "${PARAM_FILE}")
if [ $(ls -1 ${RESULT_MOVOBJ_DIR}/*_result.png 2>/dev/null | wc -l) != 0 ]; then
       ffmpeg -y -framerate ${OUT_FPS}  -pattern_type glob -i "${RESULT_MOVOBJ_DIR}/*_result.png" \
              -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${RESULT_MOVOBJ_DIR}/${RESULT_FILENAME}
       if [ $LOG_LEVEL -ge 4 ]; then
              echo "[INFO] display the final movie"
              vlc ${RESULT_MOVOBJ_DIR}/moving_object.mp4
       fi
else
       echo "[INFO] No *_result.png files found in ${RESULT_MOVOBJ_DIR}. Skipping final movie creation."
fi