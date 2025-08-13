import warnings
import typer
from pathlib import Path
import torch
from tqdm import tqdm

from cs336_basics.model import softmax, TransformerLM
from cs336_basics.training import AdamW, load_checkpoint

from tests.test_tokenizer import (
    get_tokenizer_from_vocab_merges_path,
    VOCAB_PATH,
    MERGES_PATH,
)


def _load_model(path):
    # TODO: load from stats
    params_model = dict(
        vocab_size=65536,
        context_length=256,
        d_model=512,
        d_ff=1344,
        num_layers=4,
        num_heads=16,
        rope_theta=10000,
        device=None,
    )
    model = TransformerLM(**params_model)
    params_opt = dict(
        params=model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        max_l2_norm=None,
    )
    optimizer = AdamW(**params_opt)
    _ = load_checkpoint(path, model, optimizer)
    return model.eval()


def _decode(model, ids_prefix, eot_id, temperature=0.5, top_p=0.8):
    if (n_tokens_in := len(ids_prefix)) >= model.context_length:
        warnings.warn(
            f"The prompt is too long! Context length is {model.context_length} "
            f", but got {n_tokens_in} tokens in."
        )
        return []
    max_iter = model.context_length - len(ids_prefix)
    ids_in = torch.tensor(ids_prefix)
    ids_out = []
    for _ in tqdm(range(max_iter)):
        with torch.no_grad():
            logits = model(ids_in[None]).detach().squeeze(0)
        logits = logits[-1:, :]
        if temperature == 0:
            new_token_id = logits.argmax(dim=-1).item()
        else:
            logits /= temperature
            probas = softmax(logits)
            probas, indices = probas.sort(dim=-1, descending=True)
            mask_nucleus = probas.cumsum(-1) <= top_p
            mask_nucleus[0] = True  # when the first proba > top_p
            probas, indices = probas[mask_nucleus], indices[mask_nucleus]
            idx = probas.multinomial(num_samples=1)
            new_token_id = indices[idx].item()

        if new_token_id == eot_id:
            break

        ids_out.append(new_token_id)
        ids_in = torch.cat([ids_in, torch.tensor([new_token_id])])

    return ids_out


def query(
    prompt: str,
    path_model: str = "data/tiny_stories/out/2025-08-13_16:55:05.613507/checkpoints/35.pt",
    temperature: float = 0.5,
    top_p: float = 0.8,
    eot_token="<|endoftext|>",
):
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=[eot_token]
    )
    model = _load_model(path_model)
    eot_id = tokenizer.inv_vocab[eot_token.encode("utf-8")]
    ids_in = tokenizer.encode(prompt)
    ids_out = _decode(
        model,
        ids_in,
        eot_id,
        temperature=temperature,
        top_p=top_p,
    )
    print(tokenizer.decode(ids_out))


if __name__ == "__main__":
    typer.run(query)
