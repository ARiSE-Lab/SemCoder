#!/bin/bash

##########################################################################################
# Hardware: 1x A6000 48GB GPU, or any other GPUs with at least 48GB memory
# Note: To reproduce the results reported in the paper, do not change the hyperparameters.
##########################################################################################

export CUDA_VISIBLE_DEVICES=0

OUTPUT_BASE=output_dir/eval/evalplus/
BS=16
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-base
MODEL_PATH=semcoder/semcoder_s
# extract the base name of the path as model name
model_name=$(basename $MODEL_PATH)
for DATASET in humaneval mbpp
do
    echo "Predicting: ${model_name} on ${DATASET} ..."
    
    OUTPUT_DIR=${OUTPUT_BASE}/${model_name}
    mkdir -p $OUTPUT_DIR
    OUTPUT_PATH=$OUTPUT_DIR/predictions_${DATASET}.jsonl

    # Run Inference
    python experiments/run_evalplus.py \
        --model_key $MODEL_KEY \
        --model_name_or_path $MODEL_PATH \
        --dataset $DATASET \
        --save_path $OUTPUT_PATH \
        --n_batches 1 \
        --n_problems_per_batch $BS \
        --n_samples_per_problem 1 \
        --top_p 1.0 \
        --max_new_tokens 512 \
        --temperature 0.0 \
        2>&1 | tee $OUTPUT_DIR/eval_${DATASET}.log

    # Sanitize;
    evalplus.sanitize --samples $OUTPUT_PATH;
    OUTPUT_PATH=$OUTPUT_DIR/predictions_${DATASET}-sanitized.jsonl;

    # Evaluate
    evalplus.evaluate --dataset $DATASET --samples $OUTPUT_PATH --parallel 64 --i-just-wanna-run 2>&1 | tee $OUTPUT_DIR/evalplus_${DATASET}_results.log;
done