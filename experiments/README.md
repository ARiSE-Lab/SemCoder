## Evaluation

### Get Started

- Set up [EvalPlus](https://github.com/evalplus/evalplus): HumanEval(+) and MBPP(+), as well as their evaluation utils should be already installed with [environment.yml](environment.yml)
- Set up [CRUXEval](https://github.com/facebookresearch/cruxeval):
```
git clone https://github.com/facebookresearch/cruxeval.git
```

Update the `$CRUXEVAL_HOME` to be the **absolute path** of the cloned repository in [this script](scripts/eval/eval_cruxeval.sh)
- Set up [LiveCodeBench](https://livecodebench.github.io/):
```sh
# Clone LiveCodeBench
git clone https://github.com/Robin-Y-Ding/LiveCodeBench.git; # forked version with SemCoder customization

# Set up environment
cd LiveCodeBench;
conda create -n livecodebench Python=3.10;
conda activate livecodebench;
pip install poetry;
poetry install --with with-gpu;
```

### Code Generation

- To evaluate SemCoder on EvalPlus, run
```sh
cd SemCoder;
conda activate semcoder;
# make sure you are under <path>/SemCoder/
export PYTHONPATH=$(pwd);
bash scripts/eval/eval_evalplus.sh
```
- To evaluate SemCoder on LiveCodeBench, run
```sh
cd LiveCodeBench;
conda activate livecodebench;
# make sure you are under <path>/LiveCodeBench/
bash scripts/eval/eval_codegen.sh
```

### Execution Reasoning
- To evaluate SemCoder on CRUXEval, you need to firstly clone their official release:
```sh
bash scripts/eval/eval_cruxeval.sh
```

### Rubber-duck Debugging and Self-Repair
- To finetune SemCoder for debugging and self-refinement, please refer to [this script](scripts/train/finetune_refine.sh)

- To evaluate SemCoder for iterative self-refinement on EvalPlus, please run 

```sh
bash scripts/eval/eval_finetune_refine.sh
```

- To evaluate SemCoder on [LiveCodeBench](https://livecodebench.github.io/) for code generation, please follow these steps:

```sh
# Clone our adapted LiveCodeBench
git clone https://github.com/Robin-Y-Ding/LiveCodeBench.git;

# Set up environment
cd LiveCodeBench;
conda create -n livecodebench Python=3.10;
conda activate livecodebench;
pip install poetry;
poetry install --with with-gpu;

# Run evaluation
export CUDA_VISIBLE_DEVICES=0;
python -m lcb_runner.runner.main \
  --model semcoder/semcoder_s \
  --scenario codegeneration \
  --evaluate

```
- To evaluate SemCoder on [LiveCodeBench](https://livecodebench.github.io/) for code execution, you can run:

```sh
export CUDA_VISIBLE_DEVICES=0;
cd LiveCodeBench;
python -m lcb_runner.runner.main \
    --model semcoder/semcoder_s \
    --scenario codeexecution \
    --cot_code_execution \
    --n 1 \
    --evaluate
```