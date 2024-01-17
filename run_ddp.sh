#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=1234
export WORLD_SIZE=8
export NCLL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=200

/workspace/bert/bin/torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    generate_feature.py 2>&1 | tee generate_feature.log