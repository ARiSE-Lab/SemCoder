#!/bin/bash

###################################################################################
# Hardware: 1x A6000 48GB GPU, or any other GPUs with at least 48GB memory
# Note: We use the default hyperparameters provided by the CRUXEval.
# To reproduce the results reported on the benchmark, do not change hyperparameters.
###################################################################################

export CUDA_VISIBLE_DEVICES=4

# CRUXEVAL_HOME="/proj/arise/arise/yd2447/cruxeval"
SEMCODER_HOME=$(pwd)
MODEL=semcoder/semcoder_s_1030

########################### 
# CRUXEval-I: run inference
###########################

OPT_BASE="${SEMCODER_HOME}/output_dir/eval/cruxeval/cruxeval_input"

model_name=$(basename $MODEL)_pass_1

monologue_pred_dir=${OPT_BASE}/${model_name}_monologue

mkdir -p ${monologue_pred_dir}

echo "Evaluating model: ${model_name} on CRUXEval-I with SemCoder Monologue..."

python experiments/run_cruxeval.py \
    --model $MODEL \
    --use_auth_token \
    --trust_remote_code \
    --tasks input_prediction \
    --batch_size 10 \
    --n_samples 10 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 800 \
    --temperature 0.2 \
    --save_generations \
    --save_generations_path ${monologue_pred_dir}/results.json \
    --start 0 \
    --end 800 \
    --shuffle \
    --monologue \
    --tensor_parallel_size 1

########################## 
# CRUXEval-I: Report score
##########################

echo "Reporting score for model: ${model_name}..."

python experiments/cruxeval_combine_generations.py --gen_dir ${monologue_pred_dir}
python experiments/process_cruxeval.py --task i --gen_dir ${monologue_pred_dir}

cd $CRUXEVAL_HOME/evaluation;

echo "Evaluating results: monologue prediction..."

python evaluate_generations.py \
    --generations_path ${monologue_pred_dir}/generations.json \
    --scored_results_path ${monologue_pred_dir}/scored_results.json \
    --mode input \
    2>&1 | tee ${monologue_pred_dir}/eval.log

