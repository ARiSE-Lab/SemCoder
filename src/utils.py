import os
from typing import Iterable, Sequence, TypeVar
from dataclasses import dataclass, field

N_CORES = 1 if (count := os.cpu_count()) is None or count == 0 else count // 2

@dataclass(frozen=True)
class Args:
    task: str = field(default="semcoder")
    datafile_paths: list[str] = field(default_factory=list)
    max_training_seq_length: int = field(default=1216)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=False)
    eval_dataset_size: float = field(
        default=0.05, metadata={"help": "0--1 means ratio, >1 means number of examples"}
    )
    use_flash_attention: bool = field(default=False)
    min_lr: float | None = field(default=None)

_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))

