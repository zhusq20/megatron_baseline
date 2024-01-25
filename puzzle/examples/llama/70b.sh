#! /bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_PATH=/mnt/data/zhongrx/Llama-2-70b-hf
TOKENIZER_MODEL=/mnt/data/zhongrx/Llama-2-70b-hf
DATA_PATH=/home/lijianwen/hkztmp/data-spm/redarxiv_test_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 8 \
    --micro-batch-size 4 \
    --global-batch-size 4 \
    --load-model-from-hf-config \
    --model-name-or-path $MODEL_PATH \
    --seq-length 1024 \
    --decoder-seq-length 1024 \
    --max-position-embeddings 1024 \
    --lr 0.00015 \
    --train-iters 4 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --no-position-embedding \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --fp16 \
    --recompute-activations \
"

DATA_ARGS="
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1
"
torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
