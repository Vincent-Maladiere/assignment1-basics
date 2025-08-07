from pathlib import Path
import pickle
from tqdm import tqdm
import regex as re

from cs336_basics.train_bpe import _document_parsing, PAT


class BPETokenizer:

    def __init__(self, vocab, merges, special_tokens=None, verbose=False):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.utf8_to_vocab = {i: self.inv_vocab[bytes([i])] for i in range(256)}
        self.merges = merges
        self.int_merges = {
            (self.inv_vocab[merge[0]], self.inv_vocab[merge[1]]): b"".join(merge)
            for merge in merges
        }
        self.special_tokens = special_tokens
        self.verbose = verbose

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = pickle.loads(Path(vocab_filepath).read_bytes())
        merges = pickle.loads(Path(merges_filepath).read_bytes())
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        text = self._parse_document(text)
        pretokens = self._pretokenize(text)
        del text
        ids = self._apply_merges(pretokens)
        return ids

    def encode_iterable(self, iterable):
        if isinstance(iterable, str):
            iterable = [iterable]
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids):
        buffer = []
        for token_id in ids:
            buffer.append(self.vocab[token_id])
        return b"".join(buffer).decode("utf-8", errors="replace")

    def _parse_document(self, text):
        """Split document using special tokens, and save the matches for decoding."""
        self.matches_ = []
        ordered_special_tokens = None
        if self.special_tokens is not None:
            ordered_special_tokens = sorted(
                self.special_tokens, key=lambda x: len(x), reverse=True
            )
            pattern = "|".join([re.escape(t) for t in ordered_special_tokens])
            for match_ in re.finditer(pattern, text):
                span = match_.span()
                is_outer = (span[0] == 0) or (span[1] == len(text) - 1)
                byte_token = match_.group().encode("utf-8")
                self.matches_.append(
                    dict(
                        special_token_id=self.inv_vocab[byte_token],
                        span=span,  # for debug purposes
                        is_outer=is_outer,
                    )
                )
        return _document_parsing(text, ordered_special_tokens)

    def _pretokenize(self, text):
        """Pretokenize by regex splitting."""
        pretokens = []
        for doc in text:
            doc_pretokens = []
            for pretoken in re.findall(PAT, doc):
                doc_pretokens.append(
                    # Token ids from list(pretoken.encode("utf-8")) might not match
                    # any vocab, so we have to apply an extra mapping with
                    # utf8_to_vocab.
                    tuple(
                        [self.utf8_to_vocab[x] for x in list(pretoken.encode("utf-8"))]
                    )
                )
            pretokens.append(doc_pretokens)
        return pretokens

    def _apply_merges(self, pretokens):
        """Merge token pairs without overlap between documents."""
        new_pretokens = []
        match_idx = 0
        for doc_pretokens in pretokens:
            new_pretokens += self._apply_merges_doc(doc_pretokens)
            if match_idx <= len(self.matches_) - 1:
                new_pretokens.append(self.matches_[match_idx]["special_token_id"])
                match_idx += 1

        return new_pretokens

    def _apply_merges_doc(self, pretokens):
        """Merge token pairs within a single document."""
        new_pretokens = []
        if self.verbose:
            pretokens = tqdm(pretokens)
        for pretoken in pretokens:
            # For a pretoken like " cat", loop over the byte pairs until none can be
            # found in the merges. We use a dictionnary int_merges instead of merges
            # for lookup efficiency.
            while True:
                candidates = []
                pos = 0
                while pos <= len(pretoken) - 1:
                    if (
                        token := self.int_merges.get(tuple(pretoken[pos : pos + 2]))
                    ) is not None:
                        candidates.append((self.inv_vocab[token], pos))
                    pos += 1
                if len(candidates) == 0:
                    new_pretokens += pretoken
                    break
                else:
                    token_id, pos = sorted(candidates)[0]
                    pretoken = [*pretoken[:pos], token_id, *pretoken[pos + 2 :]]

        return new_pretokens
