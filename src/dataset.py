import torch

from src.wrapper import (
    DecodingConfig,
    EncodingConfig,
    TokenizationContext,
    pad_sequences,
)
from src.utils import Args
from src.prompts import NL2CODE_PROMPT, EXEC_I_PROMPT, EXEC_O_PROMPT, REFINE_PROMPT

# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100

def map_dataset_chat(
    examples: dict[str, list[str]],
    args: "Args",
    context: TokenizationContext,
) -> dict:
    """
    Processes the multi-task samples as chat format and map into input_ids and labels
    """

    instructions = examples["instruction"]
    responses = examples["response"]

    prompts = instructions
    completions = responses

    assert len(prompts) == len(completions)
    prompt_config = EncodingConfig(add_bos=True, add_eos=False)
    completion_config = EncodingConfig(add_bos=False, add_eos=True)
    prompt_id_batches = context.encode(prompt_config, prompts)
    completion_id_batches = context.encode(completion_config, completions)

    assert len(prompt_id_batches) == len(completion_id_batches)
    # untruncated_input_ids is the "untruncated "concatenation of prompt (instructions) and completion
    untruncated_input_ids = [
        (instruction_ids + response_ids)
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    # exceeding_length is a list of bools indicating whether the a sample is truncated
    exceeding_length = [
        len(input_id) > args.max_training_seq_length
        for input_id in untruncated_input_ids
    ]
    # truncate the untruncated_input_ids to max_training_seq_length: only response/completion is truncated
    input_ids = [
        input_id[: args.max_training_seq_length] for input_id in untruncated_input_ids
    ]
    labels = [
        (list(map(lambda _: IGNORED_INDEX, instruction_ids)) + response_ids)[
            : args.max_training_seq_length
        ]
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    # `len` of each returned value must be the same, which is required by `tokenizer.map`
    # After `map`, they are treated as individual pieces of data, not as a batch.
    assert len(input_ids) == len(labels)
    for input_id_batch, label_batch in zip(input_ids, labels):
        assert len(input_id_batch) == len(label_batch)
    print(context.decode(DecodingConfig.default(), input_ids[0:])[0])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "exceeding_length": exceeding_length,
    }

def map_dataset_multitask(
    examples: dict[str, list[str]],
    args: "Args",
    context: TokenizationContext,
) -> dict:
    """
    Processes the multi-task samples into input_ids and labels
    """

    nl = examples["nl"]
    exec_simu = examples["exec_simu"]
    exec_deduc = examples["exec_deduc"]
    responses = examples["response"]

    # if args.task == "nl2code_exec":
    prompts = []
    completions = []
    for nl_i, exec_simu_i, exec_deduc_i, response_i in zip(
        nl, exec_simu, exec_deduc, responses
    ):
        # Not allow all empty samples
        assert not (nl_i == "" and exec_simu_i == "" and exec_deduc_i == "")
        # Only one of the three has content
        assert sum([nl_i != "", exec_simu_i != "", exec_deduc_i != ""]) == 1
        if nl_i != "":
            prompt = NL2CODE_PROMPT.format(instruction=nl_i, response="")
            completion = response_i
        elif exec_deduc_i != "":
            prompt = EXEC_I_PROMPT.format(instruction=exec_deduc_i, response="")
            completion = response_i
        else:
            prompt = EXEC_O_PROMPT.format(instruction=exec_simu_i, response="")
            completion = response_i
        prompts.append(prompt)
        completions.append(completion)
    # elif args.task == "nl2code_exec_refine":
    #     faults = examples["fault"]
    #     prompts = []
    #     completions = []
    #     for nl_i, exec_simu_i, exec_deduc_i, fault_i, response_i in zip(
    #         nl, exec_simu, exec_deduc, faults, responses
    #     ):
    #         # Not allow all empty samples
    #         assert not (
    #             nl_i == "" and exec_simu_i == "" and exec_deduc_i == "" and fault_i == ""
    #         )
    #         # Only one of the four has content
    #         assert sum(
    #             [nl_i != "", exec_simu_i != "", exec_deduc_i != "", fault_i != ""]
    #         ) == 1
    #         if nl_i != "":
    #             prompt = NL2CODE_PROMPT.format(instruction=nl_i, response="")
    #             completion = response_i
    #         elif exec_deduc_i != "":
    #             prompt = EXEC_I_PROMPT.format(instruction=exec_deduc_i, response="")
    #             completion = response_i
    #         elif fault_i != "":
    #             prompt = REFINE_PROMPT.format(instruction=fault_i, response="")
    #             completion = response_i
    #         else:
    #             prompt = EXEC_O_PROMPT.format(instruction=exec_simu_i, response="")
    #             completion = response_i
    #         prompts.append(prompt)
    #         completions.append(completion)
    # else:
    #     raise ValueError(f"Invalid task: {args.task}")

    assert len(prompts) == len(completions)
    prompt_config = EncodingConfig(add_bos=True, add_eos=False)
    completion_config = EncodingConfig(add_bos=False, add_eos=True)
    prompt_id_batches = context.encode(prompt_config, prompts)
    completion_id_batches = context.encode(completion_config, completions)
    assert len(prompt_id_batches) == len(completion_id_batches)
    untruncated_input_ids = [
        (instruction_ids + response_ids)
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    exceeding_length = [
        len(input_id) > args.max_training_seq_length
        for input_id in untruncated_input_ids
    ]
    input_ids = [
        input_id[: args.max_training_seq_length] for input_id in untruncated_input_ids
    ]
    labels = [
        (list(map(lambda _: IGNORED_INDEX, instruction_ids)) + response_ids)[
            : args.max_training_seq_length
        ]
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    # `len` of each returned value must be the same, which is required by `tokenizer.map`
    # After `map`, they are treated as individual pieces of data, not as a batch.
    assert len(input_ids) == len(labels)
    for input_id_batch, label_batch in zip(input_ids, labels):
        assert len(input_id_batch) == len(label_batch)
    print(context.decode(DecodingConfig.default(), input_ids[0:])[0])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "exceeding_length": exceeding_length,
    }


def get_data_collator(args: "Args", pad_token_id: int, model_key: str):
    """
    Collate a batch of examples into a batch of tensors.
    """

    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids_unpadded = [example["input_ids"] for example in examples]
        labels_unpadded = [example["labels"] for example in examples]
        padding_length = (
            args.max_training_seq_length if args.pad_to_max_length else None
        )
        if "mistral" in model_key:
            # Mistral model require left padding
            # print("Mistral model require left padding...")
            input_ids = pad_sequences(
                input_ids_unpadded, pad_token_id, "left", padding_length=padding_length
            )
            labels = pad_sequences(
                labels_unpadded, IGNORED_INDEX, "left", padding_length=padding_length
            )
        else:
            input_ids = pad_sequences(
                input_ids_unpadded, pad_token_id, "right", padding_length=padding_length
            )
            labels = pad_sequences(
                labels_unpadded, IGNORED_INDEX, "right", padding_length=padding_length
            )

        assert input_ids.shape == labels.shape
        assert len(input_ids) == len(examples)
        # Enforced in `map_raw_dataset`
        assert input_ids.shape[-1] <= args.max_training_seq_length
        if args.pad_to_max_length:
            assert input_ids.shape[-1] == args.max_training_seq_length

        if "mistral" in model_key:
            # Mistral model require left padding and requre the attnetion mask right most element to be True
            attention_mask = input_ids.ne(pad_token_id)
            # make the last element of attention_mask to be True
            attention_mask[:, -1] = True
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        else:
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": input_ids.ne(pad_token_id),
            }

    return collate
