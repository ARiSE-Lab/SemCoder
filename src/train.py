from dataclasses import dataclass
from typing import cast

from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

from src.wrapper import (
    TokenizationContext,
    get_model_context,
)
from src.utils import Args, N_CORES
from src.dataset import get_data_collator, map_dataset_chat, map_dataset_multitask

@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None



def train():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, Args))
    model_args, training_args, args = cast(
        tuple[ModelArguments, TrainingArguments, Args],
        parser.parse_args_into_dataclasses(),
    )
    if training_args.lr_scheduler_type == "cosine_with_min_lr":
        assert args.min_lr is not None
        training_args.lr_scheduler_kwargs = {
            "min_lr": args.min_lr,
        }

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    dataset = load_dataset("json", data_files=args.datafile_paths, split="train")

    model_key = model_args.model_key
    if (model_name_or_path := model_args.model_name_or_path) is None:
        model_name_or_path = model_key

    # Load tokenizer
    tokenization_context = TokenizationContext.from_model_key(
        model_key, model_name_or_path
    )
    
    # batch processing the raw examples into a format that the model can process, and save the processed examples into a Dataset instance
    if args.task == "semcoder" or args.task == "finetune_refine":
        train_dataset = dataset.map(
            function=map_dataset_multitask,
            fn_kwargs=dict(args=args, context=tokenization_context),
            batched=True,
            num_proc=N_CORES,
            remove_columns=dataset.column_names,
            load_from_cache_file= not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    elif args.task == "chat":
        train_dataset = dataset.map(
            function=map_dataset_chat,
            fn_kwargs=dict(args=args, context=tokenization_context),
            batched=True,
            num_proc=N_CORES,
            remove_columns=dataset.column_names,
            load_from_cache_file= not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    msg = f"#Examples excedding max seq length: {sum(train_dataset['exceeding_length'])} / {len(train_dataset)}"
    print(msg)

    # Shuffling
    if training_args.eval_steps is None and training_args.evaluation_strategy == "no":
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        eval_dataset = None
    else:
        print("Splitting dataset")
        split_dataset = train_dataset.train_test_split(
            test_size=args.eval_dataset_size,
            shuffle=True,
            seed=training_args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    # Load model
    state = get_model_context(
        model_key,
        model_name_or_path,
        tokenization_context,
        inference_mode=False,
        use_flash_attention=args.use_flash_attention,
    )

    print("Parallel mode:", training_args.parallel_mode)
    
    # data_collator is a function that takes a list of examples and collates them into a batch
    data_collator = get_data_collator(args, state.tokenization_context.pad_token_id, model_key=model_args.model_key)

    # wandb logging
    if "wandb" in training_args.report_to:
        import wandb
        if training_args.run_name is None:
            training_args.run_name = training_args.output_dir.split("/")[-1]
        wandb.init(project="semcoder", name=training_args.run_name, config=vars(args))

    # Initialize Huggingface Trainer: a wrapper class that handles training and evaluation
    trainer = Trainer(
        model=state.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Start Training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Training done, save the model, tokenizer, and state
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    state.tokenization_context.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()