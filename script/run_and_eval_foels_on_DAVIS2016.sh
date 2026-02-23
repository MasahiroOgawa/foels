#!/bin/bash
###

DATA_RELATIVE_DIR="foels/ext/unsupervised_detection/download/DAVIS2016/DAVIS/JPEGImages/480p"
DATASET_NAME="DAVIS2016"


source "$(dirname "$0")/run_and_eval_foels.sh"
run_and_eval_foels $DATA_RELATIVE_DIR $DATASET_NAME