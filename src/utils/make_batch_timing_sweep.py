"""Compare separate sampler/trainer GPU pools; do not change the article.

    python -m utils.make_batch_timing_sweep

NumPy and Matplotlib required. All times are predictions under the article's
roofline model. The sampler uses finite-batch simulation; training assumes
perfectly balanced work. Neither pool simulates queueing or weight updates.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from allocation import (
    HARDWARE,
    MODELS,
    _activation_bytes,
    _model_parameters,
    _statistics,
    _training_work,
    response_law,
)
from utils.make_batch_timing_preview import decode_trace, trainer_times

BUDGETS = (8, 16, 32, 64, 128)
TP_SIZES = (1, 2, 4, 8)
BATCHES = (256, 1024, 4096)
BASE = dict(
    model="Llama-3.1-8B",
    hardware="H100",
    prompt_tokens=512,
    memory_fraction=0.9,
    epochs=1,
)
OUT = Path(__file__).resolve().parents[2] / "figs/review/batch-timing-sweep"


def sampler_batches(model, hardware, maximum, tp, prompt=512, memory_fraction=0.9):
    """Two batches below both the capped-cache bound and compute boundary."""
    parameters = _model_parameters(model)
    kappa = 4 * model["L"] * model["NKV"] * model["d"]
    limit = math.floor(
        (memory_fraction * hardware["HBM"] * tp - 2 * parameters)
        / (kappa * (prompt + maximum))
    )
    upper = min(limit, math.ceil(hardware["C"] / hardware["W"]))
    return [
        (name, batch)
        for name, batch in (("half", max(1, upper // 2)), ("full", upper))
        if upper >= 1
    ]


def trainer_microbatches(
    model, hardware, maximum, n_gpus, tp, n_sequences, prompt=512, memory_fraction=0.9
):
    """Power-of-two micro-batches that fit HBM and divide this global batch."""
    ranks = n_gpus // tp
    if n_gpus % tp or n_sequences % ranks:
        return []
    available = (
        memory_fraction * hardware["HBM"] - 16 * _model_parameters(model) / n_gpus
    )
    limit = min(
        math.floor(available * tp / _activation_bytes(model, prompt + maximum)),
        n_sequences // ranks,
    )
    return [
        2**exponent
        for exponent in range(max(0, limit).bit_length())
        if 2**exponent <= limit and (n_sequences // ranks) % (2**exponent) == 0
    ]


def sampler_samples(n_gpus, tp, batch, mean_response, stats, p, seed, batches, warmup):
    """One completion trace supplies all three global batch sizes."""
    model, hw = MODELS[BASE["model"]], HARDWARE[BASE["hardware"]]
    maximum = 4 * mean_response
    replicas = n_gpus // tp
    parameters = _model_parameters(model)
    kappa = 4 * model["L"] * model["NKV"] * model["d"]
    constant = (
        max(2 * batch * parameters / hw["C"], 2 * parameters / hw["W"])
        + 8 * model["L"] * batch * model["D"] * (tp - 1) / hw["W_ici"]
    ) / tp
    kv_seconds = kappa / (tp * hw["W"])
    largest = max(BATCHES)
    recorded = batches * largest
    discarded = warmup * largest
    horizon = math.ceil(
        (discarded + recorded + max(256, batches // 8) * largest)
        * stats["mean_response"]
        / (replicas * batch)
    )
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(replicas):
        streams = []
        for _ in range(batch):
            expected = horizon / stats["mean_response"]
            size = math.ceil(expected + 10 * math.sqrt(expected) + 32)
            stream = np.minimum(rng.geometric(p, size=size), maximum)
            while stream.sum() < horizon:
                stream = np.r_[stream, np.minimum(rng.geometric(p, size=256), maximum)]
            streams.append(stream)
        traces.append(
            decode_trace(streams, horizon, BASE["prompt_tokens"], constant, kv_seconds)
        )
    times = np.concatenate([trace["times"] for trace in traces])
    times.sort()
    common_end = min(trace["end_time"] for trace in traces)
    if np.searchsorted(times, common_end, side="right") < discarded + recorded:
        raise ValueError("Replica observation windows are too short")
    samples = {}
    for nseq in BATCHES:
        boundaries = discarded + np.arange(0, recorded + 1, nseq) - 1
        samples[nseq] = np.diff(times[boundaries])
    peak_bytes = max(
        (2 * parameters + kappa * trace["peak_positions"]) / tp for trace in traces
    )
    return samples, peak_bytes


def summarize(chunks):
    values = np.concatenate(chunks)
    means = [float(np.mean(chunk)) for chunk in chunks]
    p99s = [float(np.quantile(chunk, 0.99)) for chunk in chunks]
    mean, p99 = float(np.mean(values)), float(np.quantile(values, 0.99))
    return dict(
        samples=len(values),
        mean_seconds=mean,
        p99_seconds=p99,
        p99_excess=p99 / mean - 1,
        run_mean_min=min(means),
        run_mean_max=max(means),
        run_p99_min=min(p99s),
        run_p99_max=max(p99s),
    )


def generate(out, mean_total, batches, runs, warmup, seed):
    out.mkdir(parents=True, exist_ok=True)
    model, hw = MODELS[BASE["model"]], HARDWARE[BASE["hardware"]]
    mean = mean_total - BASE["prompt_tokens"]
    maximum = 4 * mean
    if not 1 < mean or maximum + BASE["prompt_tokens"] > model["max_context"]:
        raise ValueError("Mean length must allow a four-times-mean response cap")
    p = dict(response_law("geometric", float(mean), maximum))[1]
    stats = _statistics(
        tuple(sorted(model.items())),
        "geometric",
        float(mean),
        maximum,
        BASE["prompt_tokens"],
        1.0,
    )
    parameters = _model_parameters(model)
    limit = hw["HBM"] * BASE["memory_fraction"]
    rows, case = [], 0
    for n_gpus in BUDGETS:
        for tp in TP_SIZES:
            for batch_name, batch in sampler_batches(model, hw, maximum, tp):
                chunks = {nseq: [] for nseq in BATCHES}
                peak = 0.0
                seeds = [seed + 100 * case + run for run in range(runs)]
                for run_seed in seeds:
                    samples, observed = sampler_samples(
                        n_gpus, tp, batch, mean, stats, p, run_seed, batches, warmup
                    )
                    peak = max(peak, observed)
                    for nseq in BATCHES:
                        chunks[nseq].append(samples[nseq])
                kappa = 4 * model["L"] * model["NKV"] * model["d"]
                memory = (
                    2 * parameters + kappa * batch * (BASE["prompt_tokens"] + maximum)
                ) / tp
                per_sequence = (
                    stats["mean_response"]
                    / n_gpus
                    * (
                        max(
                            2 * parameters / hw["C"], 2 * parameters / (batch * hw["W"])
                        )
                        + stats["kv_mean"] / hw["W"]
                        + 8 * model["L"] * model["D"] * (tp - 1) / hw["W_ici"]
                    )
                )
                assert memory <= limit and peak <= memory * (1 + 1e-12)
                for nseq in BATCHES:
                    result = summarize(chunks[nseq])
                    rows.append(
                        dict(
                            stage="sampler",
                            gpu_count=n_gpus,
                            tp=tp,
                            replicas_or_dp=n_gpus // tp,
                            batch_choice=batch_name,
                            batch_per_replica=batch,
                            n_sequences=nseq,
                            microbatches="",
                            memory_bytes=memory,
                            memory_budget_bytes=limit,
                            observed_peak_bytes=peak,
                            predicted_mean_seconds=nseq * per_sequence,
                            mean_relative_error=result["mean_seconds"]
                            / (nseq * per_sequence)
                            - 1,
                            seeds=seeds,
                            **result,
                        )
                    )
                case += 1
                print(
                    f"Sampler {case:2d}/40: N={n_gpus:3d}, TP={tp}, replicas={n_gpus//tp:3d}, "
                    f"B={batch:3d}; mean error {rows[-1]['mean_relative_error']:+.2%}",
                    flush=True,
                )

    # Generate IID fresh batches once and reuse their work for every trainer
    # layout. This avoids attributing independent Monte Carlo noise to scaling.
    training = {(nseq, tp): [] for nseq in BATCHES for tp in TP_SIZES}
    train_seeds = [seed + 1000000 + run for run in range(runs)]
    for run_seed in train_seeds:
        rng = np.random.default_rng(run_seed)
        stream = np.minimum(rng.geometric(p, size=batches * max(BATCHES)), maximum)
        for nseq in BATCHES:
            lengths = stream.reshape(-1, nseq)
            for tp in TP_SIZES:
                # Calculate GPU-seconds; division by each N happens below.
                layout = dict(BASE, trainer_gpus=1, trainer_tp=tp)
                training[nseq, tp].append(trainer_times(lengths, model, hw, layout))
    for n_gpus in BUDGETS:
        for tp in TP_SIZES:
            predicted = _training_work(
                model, stats, hw["C"], hw["W_ici"], tp, BASE["epochs"]
            )
            for nseq in BATCHES:
                micro = trainer_microbatches(model, hw, maximum, n_gpus, tp, nseq)
                if not micro:
                    continue
                memory = (
                    16 * parameters / n_gpus
                    + _activation_bytes(model, BASE["prompt_tokens"] + maximum) / tp
                )
                assert memory <= limit
                result = summarize([work / n_gpus for work in training[nseq, tp]])
                rows.append(
                    dict(
                        stage="trainer",
                        gpu_count=n_gpus,
                        tp=tp,
                        replicas_or_dp=n_gpus // tp,
                        batch_choice="",
                        batch_per_replica="",
                        n_sequences=nseq,
                        microbatches=micro,
                        memory_bytes=memory,
                        memory_budget_bytes=limit,
                        observed_peak_bytes="",
                        predicted_mean_seconds=nseq * predicted / n_gpus,
                        mean_relative_error=result["mean_seconds"]
                        / (nseq * predicted / n_gpus)
                        - 1,
                        seeds=train_seeds,
                        **result,
                    )
                )
    data = dict(
        base=BASE,
        mean_total_length=mean_total,
        max_response=maximum,
        gpu_counts=BUDGETS,
        tp_sizes=TP_SIZES,
        global_batch_sizes=BATCHES,
        batches_per_run_at_largest_batch=batches,
        runs=runs,
        warmup_batches_at_largest_batch=warmup,
        seed=seed,
        numpy_version=np.__version__,
        quantile_method="linear",
        rows=rows,
    )
    (out / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return data


def plot(data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "legend.frameon": False,
        }
    )
    for stage, choices, palette in (
        ("sampler", ("half", "full"), "magma"),
        ("trainer", ("",), "cividis"),
    ):
        colors = [matplotlib.colormaps[palette](v) for v in (0.12, 0.34, 0.56, 0.78)]
        fig, axes = plt.subplots(
            len(choices),
            len(BATCHES),
            figsize=(13.2, 3.4 * len(choices) + 1),
            squeeze=False,
            sharex=True,
            sharey="row",
        )
        for i, choice in enumerate(choices):
            for j, nseq in enumerate(BATCHES):
                ax = axes[i, j]
                for tp, color in zip(TP_SIZES, colors):
                    selected = sorted(
                        (
                            r
                            for r in data["rows"]
                            if r["stage"] == stage
                            and r["tp"] == tp
                            and r["n_sequences"] == nseq
                            and r["batch_choice"] == choice
                        ),
                        key=lambda r: r["gpu_count"],
                    )
                    x = [r["gpu_count"] for r in selected]
                    means, p99s = (
                        [r[key] for r in selected]
                        for key in ("mean_seconds", "p99_seconds")
                    )
                    ax.plot(x, means, color=color, lw=1.9, marker="o", ms=3)
                    ax.plot(x, p99s, color=color, lw=1.5, ls="--")
                    ax.fill_between(x, means, p99s, color=color, alpha=0.07)
                if i == 0:
                    ax.set_title(
                        rf"$N_{{\mathrm{{seq}}}}={nseq:,}$", loc="center", pad=10
                    )
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_xticks(BUDGETS, labels=[str(n) for n in BUDGETS])
                ax.grid(axis="y", color=".9", lw=0.6)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                if i == len(choices) - 1:
                    ax.set_xlabel(f"GPUs in the {stage} pool")
                if j == 0:
                    prefix = (
                        (
                            "Half batch limit\n"
                            if choice == "half"
                            else "Full batch limit\n"
                        )
                        if stage == "sampler"
                        else ""
                    )
                    ax.set_ylabel(prefix + "Batch time (seconds)")
        handles = [
            Line2D([], [], color=c, lw=2, label=f"TP{tp}")
            for c, tp in zip(colors, TP_SIZES)
        ]
        handles += [
            Line2D([], [], color=".25", lw=2, label="Mean"),
            Line2D([], [], color=".25", lw=1.6, ls="--", label="99th percentile"),
        ]
        fig.suptitle(
            f"{stage.capitalize()} batch time by GPU count, TP and batch size",
            x=0.5,
            ha="center",
            fontsize=16,
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955 if stage == "sampler" else 0.92),
            ncol=6,
            fontsize=10,
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            bottom=0.09 if stage == "sampler" else 0.17,
            top=0.87 if stage == "sampler" else 0.73,
            wspace=0.12,
            hspace=0.22,
        )
        for suffix in ("png", "svg"):
            fig.savefig(
                out / f"{stage}-timing-sweep.{suffix}", dpi=200, facecolor="white"
            )
        plt.close(fig)


def write_notes(data, out):
    model, hw = MODELS[BASE["model"]], HARDWARE[BASE["hardware"]]
    table = [
        "| TP | B: half / full limit | Replicas or DP ranks for N = 8, 16, 32, 64, 128 |",
        "|---:|---:|---|",
    ]
    for tp in TP_SIZES:
        sizes = sampler_batches(model, hw, data["max_response"], tp)
        table.append(
            f"| {tp} | {' / '.join(str(b) for _, b in sizes)} | "
            f"{', '.join(str(n // tp) for n in BUDGETS)} |"
        )
    errors = [abs(r["mean_relative_error"]) for r in data["rows"]]
    examples = {
        (r["stage"], r["n_sequences"]): r
        for r in data["rows"]
        if r["gpu_count"] == 8
        and (
            (r["stage"] == "sampler" and r["tp"] == 8 and r["batch_choice"] == "full")
            or (r["stage"] == "trainer" and r["tp"] == 1)
        )
    }
    (out / "README.md").write_text(
        f"""# GPU-count and batch-size timing sweep

These are separate stage comparisons. N means GPUs in the pool being plotted. It does not mean a shared total budget or an optimized split. The plots are not included in `main.md`.

Llama 3.1 8B, H100, mean total length {data['mean_total_length']:,} tokens, 512-token prompts. Responses follow a capped geometric law. The cap is four times the mean response length ({data['max_response']:,} tokens here); the hazard is adjusted to match the capped mean. We use BF16, one training epoch and full activation recompute. No prefill or weight sync is included.

![Sampler timing sweep](sampler-timing-sweep.png)

*Columns give the number of completed sequences per fresh batch. Colors give TP size. Solid lines are means; dashed lines are 99th percentiles. The top row uses half the feasible sampler batch limit per replica; the bottom row uses the full limit. These limits fit the capped cache within 90% of H100 memory. A compute-boundary limit is also applied; memory binds first for the default 8k workload.*

{chr(10).join(table)}

The sampler has N/TP independent replicas, each with B generators. The table gives B and the replica counts. Each replica has its own decode clock. We pool completions in wall-clock order and record intervals between successive batches of N_seq completions after warm-up. These are batch delivery intervals in a running sampler. A small interval does not mean an individual sequence finished that quickly. No queue, pacing or staleness limit is simulated.

![Trainer timing sweep](trainer-timing-sweep.png)

*The trainer has N/TP FSDP ranks, with TP/SP within each rank. Curves use independent fresh response lengths, with the same sampled batches reused across layouts. Each point is feasible with one sequence per micro-batch. `summary.csv` lists all feasible power-of-two sequence counts per micro-batch for that layout and global batch. Timing uses the article's perfectly balanced work model, including TP/SP traffic and the causal attention diagonal.*

N_seq is the whole fresh batch. It changes total work and the variation in that work. Peak activation memory depends on the micro-batch. Gradient accumulation lets the trainer use a larger whole batch without keeping all its activations in memory at once. The listed micro-batch choices have the same predicted time here: the model omits utilization differences, launch costs and FSDP communication. They can differ on GPUs.

Trainer mean and p99 both scale as 1/N at fixed TP in this model, so their ratio is unchanged by N. More GPUs alone do not smooth the randomly sampled total work. The sampler can also change its delivery fluctuations when its replica count changes. Neither plot measures trainer waiting or validates GPU performance.

At N=8, with sampler TP8 and the full batch limit, p99 is {examples['sampler', 256]['p99_excess']:.1%} above the mean for N_seq=256 and {examples['sampler', 4096]['p99_excess']:.1%} above it for N_seq=4,096. Trainer TP1 gives {examples['trainer', 256]['p99_excess']:.1%} and {examples['trainer', 4096]['p99_excess']:.1%}. Larger batches narrow the relative gap, but take proportionally longer to produce or train. Mean sequences/s stays approximately unchanged at a fixed layout in this model.

There are {sum(r['stage']=='sampler' for r in data['rows'])} sampler estimates and {sum(r['stage']=='trainer' for r in data['rows'])} trainer estimates. Each configuration has a mean, p99, memory check, rank counts and seed list in `results.json` and `summary.csv`. The largest discrepancy between a simulated mean and its analytical prediction is {max(errors):.2%}.

Each sampler layout uses {data['runs']} independent runs. Per run, we discard {data['warmup_batches_at_largest_batch']:,} batches of {max(BATCHES):,} completions, then observe {data['batches_per_run_at_largest_batch']:,} batches of that size. The same completion trace is also grouped into smaller batches. This gives {data['runs']*data['batches_per_run_at_largest_batch']:,} observations at N_seq={max(BATCHES):,}, and proportionally more at smaller N_seq. Trainer estimates use the same observation counts. Per-run p99 ranges are recorded to show Monte Carlo variation; they are not confidence intervals. Percentiles are estimates of length-induced variation under this model.

Reproduce (NumPy and Matplotlib required):

```sh
python -m utils.make_batch_timing_sweep
```

Use `--mean-total-length`, `--batches-per-run`, `--runs`, `--warmup`, `--seed` and `--output-dir` to change the experiment. The default mean length is fixed at 8k to compare GPU counts without also changing the workload. The earlier length sweep remains in `../batch-timing/`.
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--mean-total-length", type=int, default=8192)
    parser.add_argument("--batches-per-run", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    if min(args.batches_per_run, args.runs, args.warmup) < 1:
        parser.error("Run, batch and warm-up counts must be positive")
    data = generate(
        args.output_dir,
        args.mean_total_length,
        args.batches_per_run,
        args.runs,
        args.warmup,
        args.seed,
    )
    plot(data, args.output_dir)
    write_notes(data, args.output_dir)
    print(f"Wrote {len(data['rows'])} estimates and two figures to {args.output_dir}")
