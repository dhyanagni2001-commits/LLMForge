"""Serving benchmark against any OpenAI-compatible server (vLLM, SGLang).

Uses the streaming /completions endpoint. Token timing: each streamed chunk's text is counted
by the server-reported usage when available (stream_options.include_usage); TTFT is the time
to the first non-empty chunk, ITL is the gap between subsequent non-empty chunks divided by
the tokens in that chunk (1 for vLLM/SGLang streaming in practice). Actual output token
counts come from the final usage block, not from the requested max_tokens.
"""
import argparse
import asyncio
import json
import random
import statistics
import time

from benchmarks.common import environment, write


def make_prompt(n_tokens, rng):
    # ~1 token per common word for Llama-style tokenizers; actual count is recorded from usage.
    words = ["the", "model", "GPU", "memory", "kernel", "token", "batch", "cache", "layer", "data"]
    return " ".join(rng.choice(words) for _ in range(n_tokens))


async def one_request(session, url, model, prompt, max_tokens):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
               "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    chunk_times, usage = [], None
    try:
        async with session.post(f"{url}/completions", json=payload) as resp:
            if resp.status != 200:
                return {"ok": False, "status": resp.status}
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                data = json.loads(line[5:])
                if data.get("usage"):
                    usage = data["usage"]
                if data.get("choices") and data["choices"][0].get("text"):
                    chunk_times.append(time.perf_counter())
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
    end = time.perf_counter()
    if not chunk_times:
        return {"ok": False, "error": "no tokens"}
    out_tokens = (usage or {}).get("completion_tokens", len(chunk_times))
    gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    return {"ok": True, "ttft": chunk_times[0] - start, "e2e": end - start,
            "itl": gaps, "output_tokens": out_tokens,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "start": start, "end": end}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else None


async def run_level(url, model, prompts, max_tokens, concurrency):
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def bounded(p):
            async with sem:
                return await one_request(session, url, model, p, max_tokens)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(bounded(p) for p in prompts))
        wall = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]
    itl = [g for r in ok for g in r["itl"]]
    out_tokens = sum(r["output_tokens"] for r in ok)
    ms = lambda v: v * 1e3 if v is not None else None
    return {"concurrency": concurrency, "completed": len(ok), "failed": len(res) - len(ok),
            "wall_s": wall, "request_throughput": len(ok) / wall,
            "output_throughput_tok_s": out_tokens / wall,
            "ttft_p50_ms": ms(pct([r["ttft"] for r in ok], 50)),
            "ttft_p95_ms": ms(pct([r["ttft"] for r in ok], 95)),
            "itl_p50_ms": ms(pct(itl, 50)), "itl_p95_ms": ms(pct(itl, 95)),
            "e2e_p50_ms": ms(pct([r["e2e"] for r in ok], 50)),
            "e2e_p95_ms": ms(pct([r["e2e"] for r in ok], 95)),
            "mean_output_tokens": statistics.mean([r["output_tokens"] for r in ok]) if ok else 0,
            "errors": [r for r in res if not r["ok"]][:5]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--warmup-requests", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="", help="e.g. vllm-baseline, vllm-custom, sglang")
    p.add_argument("--output", default="results/serving.json")
    args = p.parse_args()

    rng = random.Random(args.seed)
    prompts = [make_prompt(args.input_tokens, rng) for _ in range(args.requests)]
    url = args.base_url.rstrip("/")
    asyncio.run(run_level(url, args.model, prompts[:args.warmup_requests], args.output_tokens, 1))
    results = {"config": vars(args), "env": environment(), "levels": []}
    for c in args.concurrency:
        lvl = asyncio.run(run_level(url, args.model, prompts, args.output_tokens, c))
        results["levels"].append(lvl)
        print(f"c={c:3d} ok={lvl['completed']} tput={lvl['output_throughput_tok_s']:.1f} tok/s "
              f"ttft_p50={lvl['ttft_p50_ms']} itl_p50={lvl['itl_p50_ms']}")
    write(args.output, results)


if __name__ == "__main__":
    main()
