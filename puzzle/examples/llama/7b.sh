#! /bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=8848
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_PATH=/home/dataset/llama-2-hf-all/Llama-2-7b-hf
TOKENIZER_MODEL=/home/dataset/llama-2-hf-all/Llama-2-7b-hf
DATA_PATH=/home/kinman/zsqtmp/data-spm/redarxiv_test_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --load-model-from-hf-config \
    --model-name-or-path $MODEL_PATH \
    --seq-length 512 \
    --decoder-seq-length 512 \
    --max-position-embeddings 512 \
    --lr 0.00015 \
    --train-iters 20 \
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
    --recompute-activations

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
    --eval-iters 1 \
    --log-timers-to-tensorboard \
    --log-world-size-to-tensorboard \
    --log-memory-to-tensorboard \
    --tensorboard-dir '/home/kinman/zsqtmp/tensorboard-log/7b_4gpu_tp2pp2'
"
torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
