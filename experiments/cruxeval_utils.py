# Reference: https://github.com/facebookresearch/cruxeval/blob/main/inference/utils.py

from math import ceil
import numpy as np
from vllm import SamplingParams
from torch.utils.data import DataLoader
import json
from abc import ABC, abstractmethod
from warnings import warn
from datasets import load_dataset, Dataset
from pprint import pprint
import math
import warnings
from collections import defaultdict
from torch.utils.data import IterableDataset
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Optional

from experiments.cruxeval_prompts import *


@dataclass
class EvalArguments:
    """
    Configuration for running the evaluation.
    """

    prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"
        },
    )
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample from the language model's output distribution."},
    )
    temperature: Optional[float] = field(
        default=0.2, metadata={"help": "Sampling temperature used for generation."}
    )
    top_k: Optional[int] = field(
        default=-1, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    eos: Optional[str] = field(
        default="<|endoftext|>", metadata={"help": "end of sentence token."}
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed used for evaluation."}
    )

class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        max_length,
        n_tasks=None,
        n_copies=1,
        prefix="",
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix

    def __iter__(self):
        prompts = []
        row_idxs = []
        for sample in range(self.n_tasks):
            dataset_sample = self.dataset[sample]
            # replace single quote to double quote: forward_monologue fine-tuning requires this
            for k in dataset_sample:
                if k != "row_index" and k != "id":
                    dataset_sample[k] = dataset_sample[k].replace("'", '"')
            prompt_contents = self.task.get_prompt(dataset_sample)
            assert isinstance(prompt_contents, str)
            prompt = self.prefix + prompt_contents
            prompts.append(prompt)
            row_idxs.append(dataset_sample["row_index"])

        return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "row_index": row_idxs[sample],
                    "prompt": prompts[sample],
                    "ids": outputs.input_ids[sample],
                    "input_len": outputs.attention_mask[sample].sum(),
                }


class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        except:
            with open(self.DATASET_PATH, "r") as f:
                lines = f.readlines()
            lines_json = [json.loads(i) for i in lines]
            data = {}
            columns = ["code", "input", "output", "id"]
            for k in columns:
                data[k] = []
            for l in lines_json:
                for k in columns:
                    data[k].append(l[k])
            data = Dataset.from_dict(data)
            self.dataset = data
            warn(
                "This task will use a locally downloaded dataset, not from the HF hub."
            )

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass



class InputPrediction(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "cruxeval-org/cruxeval"
    DATASET_NAME = None

    def __init__(self, cot = False, monologue = False):
        self.cot = cot
        self.monologue = monologue
        super().__init__(
            stop_words=["[/ANSWER]"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        # replace single quote to double quote: forward_monologue fine-tuning requires this
        doc["code"] = doc["code"].replace("'", '"')
        doc["output"] = doc["output"].replace("'", '"')
        # cot and monologue cannot be both true
        assert not (self.cot and self.monologue), "cot and monologue cannot be both true"
        if self.cot:
            return make_cot_input_prompt((doc["code"], doc["output"]))
        elif self.monologue:
            return make_backward_monologue_input_prompt((doc["code"], doc["output"]))
        else:
            return make_direct_input_prompt((doc["code"], doc["output"]))

    def get_reference(self, doc):
        return (doc["code"], doc["input"], doc["output"])

    def postprocess_generation(self, generation, idx):
        prompt = self.get_prompt(self.get_dataset()[idx])
        assert generation.startswith(prompt), print(generation, prompt)

        generation = generation[len(prompt):]
        if self.cot or self.monologue:
            if "[ANSWER]" in generation:
                generation = generation.split("[ANSWER]")[1].strip()
        if "==" in generation:
            generation = generation.split("==")[0].strip()
        if "assert" in generation:
            generation = generation.split("assert")[1].strip()
        return generation.strip()

    def process_results(self, generations, references):
        return {}
    
class OutputPrediction(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "cruxeval-org/cruxeval"
    DATASET_NAME = None

    def __init__(self, cot = False, monologue = False):
        self.cot = cot
        self.monologue = monologue
        stop_words = ["[/ANSWER]"]

        super().__init__(
            stop_words=stop_words,
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        # replace single quote to double quote: forward_monologue fine-tuning requires this
        doc["code"] = doc["code"].replace("'", '"')
        doc["input"] = doc["input"].replace("'", '"')
        # cot and monologue cannot be both true
        assert not (self.cot and self.monologue), "cot and monologue cannot be both true"
        if self.cot:
            return make_cot_output_prompt((doc["code"], doc["input"]))
        elif self.monologue:
            return make_forward_monologue_output_prompt((doc["code"], doc["input"]))
        else:
            return make_direct_output_prompt((doc["code"], doc["input"]))

    def get_reference(self, doc):
        return (doc["code"], doc["input"], doc["output"])

    def postprocess_generation(self, generation, idx):
        prompt = self.get_prompt(self.get_dataset()[idx])
        assert generation.startswith(prompt), print(generation, prompt)
        generation = generation[len(prompt):]

        if self.cot or self.monologue:
            if "[ANSWER]" in generation:
                generation = generation.split("[ANSWER]")[1].strip()
        if "==" in generation:
            generation = generation.split("==")[1].strip()
        return generation.strip()

    def process_results(self, generations, references):
        return {}

TASK_REGISTRY = {
    "input_prediction": InputPrediction,
    "output_prediction": OutputPrediction,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, cot = False, monologue = False):
    try:
        return TASK_REGISTRY[task_name](cot = cot, monologue = monologue)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")

def complete_code(
    task,
    model,
    sampling_params,
    dataloader,
    batch_size,
    n_tasks,
    log_path,
    prefix="",
    postprocess=True,
):
    max_length_generation = sampling_params.max_tokens
    code_gens = defaultdict(list)
    code_gens_raw = defaultdict(list)
    total = math.ceil(n_tasks * dataloader.dataset.n_copies)
    # TODO: check the batch > 1 case
    with open(log_path, "w") as f:
        for step, batch in tqdm(enumerate(dataloader), total=total):
            inputs = batch["ids"][:, : batch["input_len"]].tolist()
            num_tokens = len(inputs[0])
            if max_length_generation - num_tokens < 0:
                code_gens[int(batch["row_index"][0])].extend([""] * batch_size)
                code_gens_raw[int(batch["row_index"][0])].extend([""] * batch_size)
                warnings.warn(
                    f"Skipping task {batch['row_index'][0]} because it is too long -- [{max_length_generation=}|{num_tokens=}]"
                )
                continue
            sampling_params.max_tokens = max_length_generation - num_tokens
            outputs = model.generate(
                prompt_token_ids=inputs, sampling_params=sampling_params, use_tqdm=False
            )

            generated_tasks = batch["row_index"].repeat(batch_size)
            generated_texts = [o.text for o in outputs[0].outputs]
            combined_texts = [
                batch["prompt"][0] + generated_text for generated_text in generated_texts
            ]

            # write the logs of raw generations
            for task_idx, text in zip(generated_tasks, generated_texts):
                d = {}
                task_idx = int(task_idx.item())
                d["task_idx"] = task_idx
                d["prompt"] = batch["prompt"][0]
                d["response"] = text
                f.write(json.dumps(d) + "\n")
                f.flush()
                    
            for task_idx, text in zip(generated_tasks, combined_texts):
                task_idx = int(task_idx.item())
                if postprocess:
                    text_processed = task.postprocess_generation(text, task_idx)
                code_gens[task_idx].append(text_processed)
                code_gens_raw[task_idx].append(text)

    return code_gens, code_gens_raw


class Generator:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def generate(self, task_name, log_path):
        task = get_task(task_name, cot = self.args.cot, monologue = self.args.monologue)

        dataset = task.get_dataset()

        if self.args.limit is not None:
            dataset = dataset.select(range(self.args.limit))

        dataset_rows = range(dataset.num_rows)
        dataset = dataset.add_column("row_index", dataset_rows)

        if self.args.end is None:
            self.args.end = dataset.num_rows
        dataset = dataset.select(range(self.args.start, self.args.end))
        dataset_rows = range(dataset.num_rows)

        # shuffle the dataset
        if self.args.shuffle:
            dataset_rows = np.random.permutation(dataset_rows)
            dataset = dataset.select(dataset_rows)

        n_tasks = dataset.num_rows

        ds_tokenized = TokenizedDataset(
            task,
            dataset,
            self.tokenizer,
            max_length=self.args.max_length_generation,
            n_tasks=n_tasks,
            n_copies=ceil(self.args.n_samples / self.args.batch_size),
            prefix=self.args.prefix,
        )

        sampling_params = SamplingParams(
            n=self.args.batch_size,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_tokens=self.args.max_length_generation,
            stop=task.stop_words,
        )

        ds_loader = DataLoader(ds_tokenized, batch_size=1)

        generations, generations_raw = complete_code(
            task, self.model, sampling_params, ds_loader, self.args.batch_size, n_tasks, log_path
        )

        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

        if len(list(generations.values())[0]) > self.args.n_samples:
            generations = {k: v[: self.args.n_samples] for k, v in generations.items()}
            generations_raw = {k: v[: self.args.n_samples] for k, v in generations_raw.items()}
        assert all(
            [len(gen) == self.args.n_samples for gen in generations.values()]
        ), f"{[len(gen) for gen in generations.values()]}"

        return generations, generations_raw, references