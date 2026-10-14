"""Reference inference runner.

With --model: runs a Hugging Face causal LM (greedy) at a pinned revision.
Without --model: runs the built-in TinyTransformer (random fixed-seed weights) and can
swap in the custom CUDA/Triton kernels, reporting max logit difference vs the reference path.
"""
import argparse
import json

import torch


def run_hf(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 torch_dtype=dtype).to(args.device).eval()
    ids = tok(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))


def custom_ops():
    from models.transformer import Ops
    import kernels
    from kernels.triton.attention import triton_attention

    return Ops(fused_add_rmsnorm=kernels.fused_add_rmsnorm,
               attention=lambda q, k, v, causal: triton_attention(q, k, v, causal=causal))


def run_tiny(args):
    from models.transformer import Config, Ops, TinyTransformer

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    model = TinyTransformer(Config()).to(args.device, dtype).eval()
    tokens = torch.tensor([list(args.prompt.encode())[:64]], device=args.device)
    with torch.no_grad():
        ref_logits = model(tokens)
        ref_out = model.generate(tokens, args.max_new_tokens)
        result = {"reference_tokens": ref_out[0, tokens.shape[1]:].tolist()}
        if args.custom:
            model.ops = custom_ops()
            logits = model(tokens)
            out = model.generate(tokens, args.max_new_tokens)
            result.update(custom_tokens=out[0, tokens.shape[1]:].tolist(),
                          max_abs_logit_diff=(logits.float() - ref_logits.float()).abs().max().item(),
                          tokens_match=bool(torch.equal(out, ref_out)))
            model.ops = Ops()
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--revision", default="main")
    p.add_argument("--prompt", default="Explain GPU memory in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--custom", action="store_true", help="TinyTransformer: compare custom kernels")
    args = p.parse_args()
    run_hf(args) if args.model else run_tiny(args)


if __name__ == "__main__":
    main()
