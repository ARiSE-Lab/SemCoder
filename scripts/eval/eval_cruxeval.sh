#!/bin/bash

##########################################################################################
# Hardware: 1x A6000 48GB GPU, or any other GPUs with at least 48GB memory
# Note: To reproduce the results reported in the paper, do not change the hyperparameters.
##########################################################################################
export CUDA_VISIBLE_DEVICES=0

CRUXEVAL_HOME="/proj/arise/arise/yd2447/cruxeval"
SEMCODER_HOME=$(pwd)
MODEL=semcoder/semcoder_s

########################### 
# CRUXEval-I: run inference
###########################

OPT_BASE="${SEMCODER_HOME}/output_dir/eval/cruxeval/cruxeval_input"

echo "Evaluating model: ${MODEL} on CRUXEval-I (direct prediction)..."

model_name=$(basename $MODEL)
direct_pred_dir=${OPT_BASE}/${model_name}_direct

mkdir -p ${direct_pred_dir}

python experiments/run_cruxeval.py \
    --model $MODEL \
    --use_auth_token \
    --trust_remote_code \
    --tasks input_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 800 \
    --temperature 0.2 \
    --save_generations \
    --save_generations_path ${direct_pred_dir}/results.json \
    --start 0 \
    --end 800 \
    --shuffle \
    --tensor_parallel_size 1

monologue_pred_dir=${OPT_BASE}/${model_name}_monologue

mkdir -p ${monologue_pred_dir}

echo "Evaluating model: ${model_name} on CRUXEval-I with SemCoder Monologue..."

python experiments/run_cruxeval.py \
    --model $MODEL \
    --use_auth_token \
    --trust_remote_code \
    --tasks input_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 800 \
    --temperature 0.2 \
    --save_generations \
    --save_generations_path ${monologue_pred_dir}/results.json \
    --start 0 \
    --end 800 \
    --shuffle \
    --backward_monologue \
    --prompt_prefix \
    --tensor_parallel_size 1

########################## 
# CRUXEval-I: Report score
##########################

echo "Reporting score for model: ${model_name}..."

python experiments/cruxeval_combine_generations.py --gen_dir ${direct_pred_dir}
python experiments/process_cruxeval.py --task i --gen_dir ${direct_pred_dir}
python experiments/cruxeval_combine_generations.py --gen_dir ${monologue_pred_dir}
python experiments/process_cruxeval.py --task i --gen_dir ${monologue_pred_dir}

cd $CRUXEVAL_HOME/evaluation;

echo "Evaluating results: direct prediction..."

python evaluate_generations.py \
    --generations_path ${direct_pred_dir}/generations.json \
    --scored_results_path ${direct_pred_dir}/scored_results.json \
    --mode input \
    2>&1 | tee ${direct_pred_dir}/eval.log

echo "Evaluating results: monologue prediction..."

python evaluate_generations.py \
    --generations_path ${monologue_pred_dir}/generations.json \
    --scored_results_path ${monologue_pred_dir}/scored_results.json \
    --mode input \
    2>&1 | tee ${monologue_pred_dir}/eval.log

########################### 
# CRUXEval-O: run inference
###########################

cd $SEMCODER_HOME;
OPT_BASE="${SEMCODER_HOME}/output_dir/eval/cruxeval/cruxeval_output"

echo "Evaluating model: ${MODEL} on CRUXEval-O (direct prediction)..."

model_name=$(basename $MODEL)
direct_pred_dir=${OPT_BASE}/${model_name}_direct

mkdir -p ${direct_pred_dir}

python experiments/run_cruxeval.py \
    --model $MODEL \
    --use_auth_token \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 800 \
    --temperature 0.2 \
    --save_generations \
    --save_generations_path ${direct_pred_dir}/results.json \
    --start 0 \
    --end 800 \
    --shuffle \
    --tensor_parallel_size 1

monologue_pred_dir=${OPT_BASE}/${model_name}_monologue

mkdir -p ${monologue_pred_dir}

echo "Evaluating model: ${model_name} on CRUXEval-O with SemCoder Forward Monologue..."

python experiments/run_cruxeval.py \
    --model $MODEL \
    --use_auth_token \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 800 \
    --temperature 0.2 \
    --save_generations \
    --save_generations_path ${monologue_pred_dir}/results.json \
    --start 0 \
    --end 800 \
    --shuffle \
    --forward_monologue \
    --prompt_prefix \
    --annotate_src \
    --tensor_parallel_size 1

########################## 
# CRUXEval-O: Report score
##########################
echo "Reporting score for model: ${model_name}...";

python experiments/cruxeval_combine_generations.py --gen_dir ${direct_pred_dir}
python experiments/process_cruxeval.py --task o --gen_dir ${direct_pred_dir}
python experiments/cruxeval_combine_generations.py --gen_dir ${monologue_pred_dir}
python experiments/process_cruxeval.py --task o --gen_dir ${monologue_pred_dir}

cd $CRUXEVAL_HOME/evaluation;

echo "Evaluating results: direct prediction..."

python evaluate_generations.py \
    --generations_path ${direct_pred_dir}/generations.json \
    --scored_results_path ${direct_pred_dir}/scored_results.json \
    --mode output \
    2>&1 | tee ${direct_pred_dir}/eval.log

echo "Evaluating results: monologue prediction..."

python evaluate_generations.py \
    --generations_path ${monologue_pred_dir}/generations.json \
    --scored_results_path ${monologue_pred_dir}/scored_results.json \
    --mode output \
    2>&1 | tee ${monologue_pred_dir}/eval.log

