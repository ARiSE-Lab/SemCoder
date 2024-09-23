import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from src.wrapper import GenerationConfig, get_model_context
from src.prompts import REFINE_PROMPT, REFINE_PROMPT_ONE_SHOT
from src.utils import chunked
import json

PROMPT_TEMPLATE = """<Prompt>
{prompt}
<Faulty Trace>
```python
{trace}
```
<Failed Test>
```python
{failed_test}
```"""

class NL2CodeProblem(TypedDict):
    id: str
    instruction: str
    response_prefix: str


def get_mbpp_raw_problems() -> list[dict]:
    problems = get_mbpp_plus()
    return list(problems.values())


def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())


def map_problem(p: dict) -> NL2CodeProblem:
    id = p["raw_index"]
    instruction = PROMPT_TEMPLATE.format(prompt=p["nl"], trace=p["buggy_trace"], failed_test=p["failed_test"])
    response_prefix = f"""The root cause of the failure"""
    
    return NL2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )

def map_problem_one_shot(p: dict) -> NL2CodeProblem:
    id = p["raw_index"]
    instruction = PROMPT_TEMPLATE.format(prompt=p["nl"], trace=p["buggy_trace"], failed_test=p["failed_test"])
    response_prefix = f"""The root cause of the failure is that """
    
    return NL2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )

@dataclass(frozen=True)
class Args:
    model_key: str
    dataset: Literal["humaneval", "mbpp"]
    save_path: str
    fault_history_file: str

    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int
    one_shot: bool

    model_name_or_path: str | None = None


def main():
    parser = HfArgumentParser((Args, GenerationConfig))
    args, generation_config = cast(
        tuple[Args, GenerationConfig],
        parser.parse_args_into_dataclasses(),
    )
    if args.one_shot:
        raw_problem_fn, map_problem_fn = (
            (get_humaneval_raw_problems, map_problem_one_shot)
            if args.dataset == "humaneval"
            else (get_mbpp_raw_problems, map_problem_one_shot)
        )
    else:
        raw_problem_fn, map_problem_fn = (
            (get_humaneval_raw_problems, map_problem)
            if args.dataset == "humaneval"
            else (get_mbpp_raw_problems, map_problem)
        )
    raw_problems = raw_problem_fn()
    # map the raw problems to a dict {"<task_id>": <prompt>}"
    raw_problems_dict = {p["task_id"]: p["prompt"] for p in raw_problems}

    with open(args.fault_history_file, "r") as f:
        fault_history = f.readlines()

    fault_history = [json.loads(fault) for fault in fault_history]

    problems = list(map(map_problem_fn, fault_history))

    state = get_model_context(args.model_key, args.model_name_or_path)

    problems_chunked = list(chunked(list(problems), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches

    Path(args.save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        if args.one_shot:
            prompts = [
                REFINE_PROMPT_ONE_SHOT.format(
                    instruction=problem["instruction"], response=problem["response_prefix"]
                )
                for problem in problems
            ]
        else:
            prompts = [
                REFINE_PROMPT.format(
                    instruction=problem["instruction"], response=problem["response_prefix"]
                )
                for problem in problems
            ]
        print("PROMPT")
        print(prompts[-1])
        all_prompts = prompts * args.n_samples_per_problem
        all_task_ids = task_ids * args.n_samples_per_problem
        response = state.complete(generation_config, all_prompts)
        completions = response.decoded_outputs
        assert len(problems) <= args.n_problems_per_batch
        assert len(completions) == len(problems) * args.n_samples_per_problem
        print("COMPLETION")
        print(completions[-1])
        samples = []
        for task_id, completion in zip(all_task_ids, completions):
            if "```python" in completion:
                start_idx = completion.find("```python") + len("```python")
            else:
                start_idx = 0
            if "```" in completion[start_idx + 1:]:
                end_idx = completion.find("```", start_idx + 1)
            else:
                end_idx = len(completion)
            samples.append(
                dict(
                    task_id=task_id,
                    completion=completion[start_idx:end_idx],
                )
            )
        for sample in samples:
            sample["solution"] = sample["completion"]
        write_jsonl(args.save_path, samples, append=True)


if __name__ == "__main__":
    main()
