# 🗣️ SemCoder: Training Code Language Models with Comprehensive Semantics

<p align="center">
    <a href="https://arxiv.org/abs/2406.01006"><img src="https://img.shields.io/badge/arXiv-2406.01006-b31b1b.svg?style=for-the-badge">
</p>

<p align="center">
    🤖&nbsp;<a href="#-models">Models</a>
    | 🛠️&nbsp;<a href="#-get-started">Get Started</a>
    | 🕹️&nbsp;<a href="#-demo">Demo</a>
    | 📝&nbsp;<a href="#-citation">Citation</a>
    | 🙏&nbsp;<a href="#-acknowledgements">Acknowledgements</a>
</p>

> [!NOTE]
> 
> __Work in Progress__: The repository is still work in progress. We are targeting to finalize the release by the end of October, 2024. Stay Tuned!


## 🤖 Models

| Model      | Checkpoint                                               | Size | License                                                                           |
|------------|----------------------------------------------------------|------|-----------------------------------------------------------------------------------|
| SemCoder   | 🤗 [HF Link](https://huggingface.co/semcoder/semcoder)   | 6.7B | [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL) |
| SemCoder-S | 🤗 [HF Link](https://huggingface.co/semcoder/semcoder_s) | 6.7B | [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL) |


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
