#!/bin/bash

config=${1:-configs/cost/costb_r50_8x8x1_48e_mit_rgb.py}
gpus=${2:-0,1,2,3,4,5,6,7}
seed=${3:-0}

ngpus=$(echo $gpus | awk -F',' '{ print NF }')

set -x
if [ $ngpus -gt 1 ]; then
    port=$(shuf -i 49152-65535 -n 1)  # random port
    CUDA_VISIBLE_DEVICES=$gpus PORT=$port \
        tools/dist_train.sh $config $ngpus \
        --validate --seed $seed --deterministic ${@:4}
else
    CUDA_VISIBLE_DEVICES=$gpus \
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        python tools/train.py $config --gpus 1 \
        --validate --seed $seed --deterministic ${@:4}
fi
