#!/bin/bash

##########################################################################################
# Hardware: 1x A6000 48GB GPU, or any other GPUs with at least 48GB memory
# Note: To reproduce the results reported in the paper, do not change the hyperparameters.
##########################################################################################

export CUDA_VISIBLE_DEVICES=0

# RUN_NAME and MODEL_PATH should be the same as the one used in the scripts/train/finetune_refine.sh
RUN_NAME=finetune_semcoder_s_refine
MODEL_PATH=output_dir/${RUN_NAME}

OUTPUT_BASE=output_dir/eval/evalplus_refine/
BS=16
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-base

############################## 
# First-time NL2Code Inference
##############################

for DATASET in humaneval mbpp
do 
    model_name=$RUN_NAME/checkpoint-last
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
    OUTPUT_PATH=$OUTPUT_DIR/predictions_${DATASET}-sanitized.jsonl

    # Evaluate
    evalplus.evaluate --dataset $DATASET --samples $OUTPUT_PATH --parallel 64 --i-just-wanna-run 2>&1 | tee $OUTPUT_DIR/evalplus_${DATASET}_results.log
done

########################### 
# Iterative Self-Refinement
###########################

for DATASET in humaneval mbpp
do  
    FAULTY_FILE_BASE=${OUTPUT_BASE}/${RUN_NAME}/checkpoint-last
    for REFINE in 1 2 3 4 5 # we allow several rounds of refinement
    do
        model_name=$RUN_NAME/refine-${REFINE}
        OUTPUT_DIR=${OUTPUT_BASE}/${model_name}
        mkdir -p $OUTPUT_DIR

        echo "Tracing for ${REFINE}th time the faulty predictions for ${model_name} on ${DATASET} ..."
        FAULTY_EVAL_RES=$FAULTY_FILE_BASE/predictions_${DATASET}-sanitized_eval_results.json
        FAULTY_TRACE=$OUTPUT_DIR/faulty_predictions_traces_${DATASET}.jsonl
        rm -rf $FAULTY_TRACE;
        python experiments/trace_evalplus.py \
            --eval_result_file $FAULTY_EVAL_RES \
            --output_file $FAULTY_TRACE

        echo "Processing the faulty traces for refinement"
        FAULTY_PROCESSED=$OUTPUT_DIR/faulty_predictions_${DATASET}_processed.jsonl
        python experiments/build_self_refine_input.py \
            --bug_report_file $FAULTY_TRACE \
            --inference \
            --output_file $FAULTY_PROCESSED


        echo "Refining for the ${REFINE}th time: ${model_name} on ${DATASET} with faulty history file of ${FAULTY_PROCESSED} ..."
        # Run Refinement Inference
        OUTPUT_PATH=$OUTPUT_DIR/predictions_${DATASET}.jsonl
        
        if [ $DATASET == "humaneval" ]; then
            REFINE_TEMP=0.6
        else
            REFINE_TEMP=0.8
        fi
        python experiments/run_refine_evalplus.py \
            --model_key $MODEL_KEY \
            --model_name_or_path $MODEL_PATH \
            --dataset $DATASET \
            --save_path $OUTPUT_PATH \
            --n_batches 1 \
            --n_problems_per_batch $BS \
            --n_samples_per_problem 1 \
            --top_p 0.95 \
            --max_new_tokens 512 \
            --temperature $REFINE_TEMP \
            --fault_history_file $FAULTY_PROCESSED \
            2>&1 | tee $OUTPUT_DIR/eval_refine_${DATASET}.log

        echo "Merge ..."
        # Merge the original prediction file with the refined prediction file
        ORIG_PRED_FILE=$FAULTY_FILE_BASE/predictions_${DATASET}-sanitized.jsonl
        python experiments/merge_refine_for_eval.py \
            --orig_pred_file $ORIG_PRED_FILE \
            --refine_pred_file $OUTPUT_PATH \
            --output_file $OUTPUT_DIR/predictions_${DATASET}-merged.jsonl

        echo "Sanitize ..."
        # Sanitize;
        evalplus.sanitize --samples $OUTPUT_DIR/predictions_${DATASET}-merged.jsonl;
        OUTPUT_PATH=$OUTPUT_DIR/predictions_${DATASET}-merged-sanitized.jsonl

        echo "Evaluate ..."
        # Evaluate
        evalplus.evaluate --dataset $DATASET --samples $OUTPUT_PATH --parallel 64 --i-just-wanna-run 2>&1 | tee $OUTPUT_DIR/evalplus_${DATASET}_results.log
        mv $OUTPUT_PATH $OUTPUT_DIR/predictions_${DATASET}-sanitized.jsonl
        mv $OUTPUT_DIR/predictions_${DATASET}-merged-sanitized_eval_results.json $OUTPUT_DIR/predictions_${DATASET}-sanitized_eval_results.json # renaming for next iteration
        FAULTY_FILE_BASE=$OUTPUT_DIR
    done
done