import pickle
import regex as re
from itertools import islice
from functools import partial
from pathlib import Path
from collections import Counter
from joblib import Parallel, delayed, effective_n_jobs
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def fast_line_count(filename):
    """Use the raw buffer interface to load byte chunks faster and count lines."""
    with open(filename, "rb") as f:
        bufgen = iter(partial(f.raw.read, 1024 * 1024), b"")
        return sum(buf.count(b"\n") for buf in bufgen)


def _load_chunks(path_input, n_lines, rank, world_size):
    """Each worker loads a chunk corresponding to its rank and the world size."""
    if not (path_input := Path(path_input)).exists():
        raise FileNotFoundError(str(path_input))

    if world_size == 1:
        return path_input.read_text()

    chunk_size = n_lines // world_size
    start, end = chunk_size * rank, chunk_size * (rank + 1)

    with open(path_input, "r") as file:
        # isslice allows fast read access to specific lines.
        text = "".join(islice(file, start, end))

    return text


def _document_parsing(text, special_tokens):
    if special_tokens:
        text = re.split("|".join([re.escape(t) for t in special_tokens]), text)
    else:
        text = [text]
    return text


def _get_pretoken_freq(text):
    pretoken_freq = Counter()
    for doc in text:
        for pretoken in re.findall(PAT, doc):
            pretoken_freq[tuple(pretoken.encode("utf-8"))] += 1
    return pretoken_freq


def _get_pair_counts(pretoken_freq):
    pair_counts = Counter()
    for pretoken, freq in pretoken_freq.items():
        for pos in range(len(pretoken) - 1):
            pair = (pretoken[pos], pretoken[pos + 1])
            pair_counts[pair] += freq
    return pair_counts


def _load_and_pretokenize(path_input, n_lines, rank, world_size, special_tokens):
    text = _load_chunks(path_input, n_lines, rank, world_size)
    text = _document_parsing(text, special_tokens)
    pretoken_freq = _get_pretoken_freq(text)
    pair_counts = _get_pair_counts(pretoken_freq)
    return pair_counts, pretoken_freq


def _get_most_common(pair_counts, vocab):
    """Return the most common pair, and use the lexigraphic order when there are ties."""
    pair, count = pair_counts.most_common(1)[0]
    top = 1
    # We fetch all the ties having the top count, by growing progressively the candidate
    # pool, since collection.Counter returns top items in an arbitrary order.
    while pair_counts.most_common(top + 1)[-1][1] == count:
        top += 1
    if top == 1:
        pair, count

    def to_vocab(pair):
        return vocab[pair[0]], vocab[pair[1]]

    return sorted(
        pair_counts.most_common(top), key=lambda x: to_vocab(x[0]), reverse=True
    )[0]


def _make_new_pretoken(pretoken, pair, new_token_id):
    """Create the post-merged pretoken and its overlapping pairs.

    This function also handle the edge-case when there are several merges in the same
    pretoken, e.g. "(a, b, c, a, b)" when merging the pair "(a, b)".
    """
    # First, compute the final new pretoken, after all merges, and identify the
    # prefixes and suffixes before merging. Later, these pre-merging suffixes and
    # prefixes will allow to decrease the count of pre-merging pairs.
    new_pretoken = []
    idx = 0
    prefix_suffix = []
    while idx <= len(pretoken) - 1:
        if pretoken[idx : idx + 2] == pair:
            new_pretoken.append(new_token_id)
            prefix, suffix = pretoken[:idx], pretoken[idx + 2 :]
            prefix_suffix.append((prefix, suffix))
            idx += 2
        else:
            new_pretoken.append(pretoken[idx])
            idx += 1

    # Now that we have our merged new_pretoken, we can substitute it to get our
    # post-merge prefixes and suffixes. We will use these to increase the count
    # of post-merging pairs.
    results = []
    new_pretoken = tuple(new_pretoken)
    idx = 0
    while idx <= len(new_pretoken) - 1:
        if new_pretoken[idx] == new_token_id:
            new_prefix, new_suffix = new_pretoken[:idx], new_pretoken[idx + 1 :]
            new_res = (
                new_pretoken,
                *prefix_suffix[len(results)],
                new_prefix,
                new_suffix,
            )
            results.append(new_res)
        idx += 1

    return results


class BPETrainer:

    def __init__(
        self,
        path_input,
        vocab_size,
        special_tokens,
        path_output=None,
        n_jobs=None,
        verbose=False,
    ):
        self.path_input = path_input
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.path_output = path_output
        self.n_jobs = n_jobs
        self.verbose = verbose

    def train(self):
        """Train the BPE encoder on a document and save vocab and merges on disk."""

        n_lines = fast_line_count(self.path_input)
        world_size = effective_n_jobs(self.n_jobs)

        # Each worker loads a section of the document and perform pretokenization and
        # counting. Joblib is an alternative to the multiprocess package to implement
        # this embarrassingly parallel loop.
        pair_counts, pretoken_freq = Counter(), Counter()
        parallel = Parallel(n_jobs=self.n_jobs, verbose=1, return_as="generator")
        for pair_counts_, pretoken_freq_ in parallel(
            delayed(_load_and_pretokenize)(
                self.path_input, n_lines, rank, world_size, self.special_tokens
            )
            for rank in range(world_size)
        ):
            pair_counts += pair_counts_
            pretoken_freq += pretoken_freq_

        if self.verbose:
            print("Pre-tokenization done")

        self.vocab_ = {i: bytes([i]) for i in range(256)} | {
            256 + i: t.encode("utf-8") for i, t in enumerate(self.special_tokens)
        }
        self.merges_ = []

        pbar = tqdm(total=self.vocab_size - len(self.vocab_))
        while len(self.vocab_) < self.vocab_size:

            int_pair, top_freq = _get_most_common(pair_counts, self.vocab_)
            if top_freq == 1:
                break
            byte_pair = (self.vocab_[int_pair[0]], self.vocab_[int_pair[1]])
            new_token = b"".join(byte_pair)
            new_token_id = len(self.vocab_)
            self.vocab_[new_token_id] = new_token
            self.merges_.append(byte_pair)
            pair_counts, pretoken_freq = self._replace_most_frequent_pair(
                new_token_id,
                int_pair,
                pair_counts,
                pretoken_freq,
            )
            if self.verbose:
                print(new_token_id, new_token, top_freq)
            pbar.update(1)

        if self.path_output is not None:
            self._save()

        return self

    def _replace_most_frequent_pair(
        self, new_token_id, int_pair, pair_counts, pretoken_freq
    ):
        """Update pair_counts and pretoken_freq with new_token_id."""
        new_pretoken_freq = Counter()
        for pretoken, freq in pretoken_freq.items():
            new_pretoken = None
            for (
                new_pretoken,
                prefix,
                suffix,
                new_prefix,
                new_suffix,
            ) in _make_new_pretoken(pretoken, int_pair, new_token_id):
                if prefix:
                    left_pair = (prefix[-1], int_pair[0])
                    new_left_pair = (new_prefix[-1], new_token_id)
                    # The condition below prevents counting the pair
                    # (new_token_id, new_token_id) twice.
                    if new_prefix[-1] != new_token_id:
                        pair_counts[left_pair] -= freq
                        pair_counts[new_left_pair] += freq

                if suffix:
                    right_pair = (int_pair[1], suffix[0])
                    new_right_pair = (new_token_id, new_suffix[0])
                    pair_counts[right_pair] -= freq
                    pair_counts[new_right_pair] += freq

                # Can be run several time, as new_pretoken is invariant in this
                # current loop.
                new_pretoken_freq[new_pretoken] = freq

            if new_pretoken is None:
                new_pretoken_freq[pretoken] = freq

        pair_counts.pop(int_pair)

        return pair_counts, new_pretoken_freq

    def _save(self):
        self.path_output = Path(self.path_output)
        self.path_output.mkdir(parents=True, exist_ok=True)

        path_vocab = self.path_output / "vocab.pkl"
        path_vocab.write_bytes(pickle.dumps(self.vocab_))
        print(f"Wrote {path_vocab}")

        path_merges = self.path_output / "merges.pkl"
        path_merges.write_bytes(pickle.dumps(self.merges_))
        print(f"Wrote {path_merges}")
