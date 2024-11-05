# 🤔 SemCoder: Training Code Language Models with Comprehensive Semantics

<p align="center">
    <a href="https://arxiv.org/abs/2406.01006"><img src="https://img.shields.io/badge/arXiv-2406.01006-b31b1b.svg?style=for-the-badge">
</p>

<p align="center">
    🤔&nbsp;<a href="#-overview">Overview</a>
    | 🤖&nbsp;<a href="#-models">Models</a>
    | 📚&nbsp;<a href="#-dataset">Dataset</a>
    | 🛠️&nbsp;<a href="#-get-started">Get Started</a>
    | 🕹️&nbsp;<a href="#-demo">Demo</a>
    | 📝&nbsp;<a href="#-citation">Citation</a>
    | 🙏&nbsp;<a href="#-acknowledgements">Acknowledgements</a>
</p>

## 📰 News

- []

## 🤔 Overview

## 🤖 Models

| Model      | HF Ckpt                                               | Size | HEval (+)   | MBPP (+)    | LCB-Lite    | CXEval-I  | CXEval-O  | LCB-Ex |
|------------|----------------------------------------------------------|------|-------------|-------------|-------------|-------------|-------------|-------------|
| SemCoder   | 🤗 [HF Link](https://huggingface.co/semcoder/semcoder_1030)   | 6.7B | --.- (--.-) | --.- (--.-) | --.- | --.- | --.- | --.- |
| SemCoder-S | 🤗 [HF Link](https://huggingface.co/semcoder/semcoder_s) | 6.7B | --.- (--.-) | --.- (--.-) | --.- | --.- | --.- | --.- |

## 📚 Dataset

* [PyX](?): ???
* [PyX-Monologue](?): ???.
* [PyX-R](?): ???.

## 🛠️ Get Started

### Install Environment
```sh
git clone https://github.com/ARiSE-Lab/SemCoder.git;
cd SemCoder;
conda env create --name semcoder --file=environment.yml;
conda activate semcoder;
export PYTHONPATH=$(pwd);
```

### 🕹️ Demo

We follow [Magicoder](https://github.com/ise-uiuc/magicoder/blob/main/demo/magicoder_demo.py) script to lanuch a gradio server for the local demo. You can launch your local gradio demo as following:

```bash
CUDA_VISIBLE_DEVICES=0 python semcoder_demo.py \
   --base_model "semcoder/semcoder_s" \
   --device "cuda:0" \
   --port 8080
```


### Evaluation


- To evaluate SemCoder on [EvalPlus](https://github.com/evalplus/evalplus), run
```sh
bash scripts/eval/eval_evalplus.sh
```

- To evaluate SemCoder on [CRUXEval](https://github.com/evalplus/evalplus), you need to firstly clone their official release:

```
git clone https://github.com/facebookresearch/cruxeval.git
```

Update the `$CRUXEVAL_HOME` to be the **absolute path** of the cloned repository in [this script](scripts/eval/eval_cruxeval.sh) and run:

```sh
bash scripts/eval/eval_cruxeval.sh
```

- To finetune SemCoder for debugging and self-refinement, please refer to [this script](scripts/train/finetune_refine.sh)

- To evaluate SemCoder for iterative self-refinement on EvalPlus, please run 

```sh
bash scripts/eval/eval_finetune_refine.sh
```

- ✨ __[New]__ ✨ To evaluate SemCoder on [LiveCodeBench](https://livecodebench.github.io/) for code generation, please follow these steps:

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
- ✨ __[New]__ ✨ To evaluate SemCoder on [LiveCodeBench](https://livecodebench.github.io/) for code execution, you can run:

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

## 📝 Citation

```bibtex
@article{ding2024semcoder,
  title={SemCoder: Training Code Language Models with Comprehensive Semantics},
  author={Yangruibo Ding and Jinjun Peng and Marcus J. Min and Gail Kaiser and Junfeng Yang and Baishakhi Ray},
  journal={arXiv preprint arXiv:2406.01006},
  year={2024}
}
```

## 🙏 Acknowledgements

We thank the following amazing projects that inspired our design choices:

- [MagicCoder](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder): Synthetic Code Generation.
- [EvalPlus](https://github.com/evalplus/evalplus): Test-case Generation & Augmentation.
- [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder): Base model for SemCoder.
