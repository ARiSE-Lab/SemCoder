#!/bin/bash

##########################################################################################
# Hardware: 8x A6000 48GB GPU, or any other GPUs with at least 48GB memory
# Note: To reproduce the results reported in the paper, do not change the hyperparameters.
##########################################################################################

BS=16
GRAD_ACC=4
LR=1e-5
EVAL_STEPS=50
SAVE_STEPS=1000
SAVE_TOTAL=1
LOGGING_STEPS=2
EPOCH=2
WARMUP=5
MODEL=semcoder/semcoder_s
DATA_FILE=None # TODO: Add the path to the data file

RUN_NAME=finetune_semcoder_s_refine
OUTPUT_DIR=output_dir/${RUN_NAME}
mkdir -p $OUTPUT_DIR

deepspeed --master_port 29700 --include="localhost:0,1,2,3,4,5,6,7" src/train.py \
    --task finetune_refine \
    --model_name_or_path $MODEL \
    --model_key deepseek-ai/deepseek-coder-6.7b-base \
    --datafile_paths $DATA_FILE \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL \
    --learning_rate $LR \
    --warmup_steps $WARMUP \
    --logging_steps $LOGGING_STEPS \
    --lr_scheduler_type cosine \
    --overwrite_cache true \
    --use_flash_attention true \
    --deepspeed config/deepspeed_config.json \
    --report_to wandb \
    --fp16 true \
    --max_training_seq_length 2048 \
    --gradient_checkpointing true \
    2>&1 | tee $OUTPUT_DIR/train.log
