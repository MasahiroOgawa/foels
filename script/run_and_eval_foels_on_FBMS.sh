#!/bin/bash
###

DATA_RELATIVE_DIR="foels/ext/unsupervised_detection/download/FBMS/Testset"
DATASET_NAME="FBMS"
RESULT_DIR="FBMS/Testset"


source "$(dirname "$0")/run_and_eval_foels.sh"
run_and_eval_foels $DATA_RELATIVE_DIR $DATASET_NAME $RESULT_DIR


