#!/bin/bash
###

run_and_eval_foels(){
    # Arguments are passed positionally
    local data_relative_dir="$1" # Use local for function-scoped variables
    local dataset_name="$2"
    local result_dir="${3:-${dataset_name}}" # result directory name relative to roo/result/. Default to dataset_name if not provided

    # stop immediately when error occurred
    set -eu

    echo "[INFO] set parameters."
    ROOT_DIR=$(dirname "$0")/..
    ROOT_DATA_DIR=${ROOT_DIR}/${data_relative_dir}
    DATASET_NAME=${dataset_name}
    RESULT_DIR=${ROOT_DIR}/result/${result_dir}
    
    echo "[INFO] check input file existence."
    if [ -n "$(ls -A ${ROOT_DATA_DIR})" ]; then
        echo "[INFO] input files exist. skip downloading."
    else
        echo "[INFO] input files do not exist. set up and download it."
        (
            cd ${ROOT_DIR}/foels/ext/unsupervised_detection
            # to avoid "ADDR2LINE: unbound variable" error, unset -u.
            set +eu
            eval "$(conda shell.bash activate contextual-information-separation)"
            set -eu
            echo "[INFO] env: $CONDA_DEFAULT_ENV"
            python ./scripts/test_${DATASET_NAME}_raw.py
        )
    fi


    echo "[INFO] run foels on the dataset."
    for DATA_DIR in ${ROOT_DATA_DIR}/*
    do
        echo "[INFO] read DATA_DIR=${DATA_DIR}"
        $ROOT_DIR/script/run_foels.sh $DATA_DIR ${RESULT_DIR}
    done

    echo "[INFO] run evaluation."
    echo "[INFO] deactivate current uv venv first. otherwise, conda env will be hide."
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        echo "[INFO] No virtualenv activated."
    else
        echo "[INFO] deactivate current uv venv: $VIRTUAL_ENV"
        PATH=$(echo $PATH | tr ':' '\n' | grep -v "$VIRTUAL_ENV" | tr '\n' ':')
        unset VIRTUAL_ENV
    fi
    (
        cd ${ROOT_DIR}/foels/ext/unsupervised_detection
        # to avoid "ADDR2LINE: unbound variable" error, unset -u.
        set +eu
        eval "$(conda shell.bash activate contextual-information-separation)"
        set -eu
        echo "[INFO] env: $CONDA_DEFAULT_ENV"
        python ./scripts/test_${DATASET_NAME}_foels.py
    )

    echo "[INFO] copy results to result directory."
    cp ${ROOT_DIR}/foels/ext/unsupervised_detection/results/Foels/${result_dir}/result.csv ${RESULT_DIR}

    echo "[INFO] finish."
}