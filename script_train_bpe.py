import numpy as np
from pathlib import Path

from tests.test_tokenizer import (
    get_tokenizer_from_vocab_merges_path,
    VOCAB_PATH,
    MERGES_PATH,
)


tokenizer = get_tokenizer_from_vocab_merges_path(
    VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
)

output_path = Path("data/tiny_stories/ids")
output_path.mkdir(exist_ok=True, parents=True)

buffer_size = 1_000_000
buffer = np.empty((buffer_size), dtype="uint16")
with open("data/TinyStoriesV2-GPT4-train.txt") as f:
    for i, token_id in enumerate(tokenizer.encode_iterable(f)):
        if i > 0 and i % buffer_size == 0:
            path_file = output_path / str(i)
            np.save(path_file, buffer)
            print(f"Wrote {path_file}")
            buffer = np.empty((buffer_size), dtype="uint16")
        buffer[i % buffer_size] = token_id
