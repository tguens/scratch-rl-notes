"""Finite-batch timing previews, kept outside the article.

Run: python -m utils.make_batch_timing_preview
Requires NumPy and Matplotlib. Solvers and article figures are unchanged.
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

LAYOUT = dict(
    model="Llama-3.1-8B",
    hardware="H100",
    prompt_tokens=512,
    n_sequences=1024,
    sampler_tp=8,
    replicas=3,
    batch_per_replica=32,
    trainer_gpus=8,
    trainer_tp=1,
    epochs=1,
    memory_fraction=0.9,
)
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)
OUT = Path(__file__).resolve().parents[2] / "figs/review/batch-timing"


def decode_trace(lengths, horizon, prompt, constant, kv_seconds):
    """Exact affine step-time sum, evaluated only at completion events.

    `lengths` gives one response-length stream per generator. `constant` is
    linear-layer plus TP seconds per step; `kv_seconds` is seconds per cached
    token position. All generators start empty and immediately replace EOS.
    """
    batch = len(lengths)
    ends, completed = [], []
    for stream in lengths:
        stream = np.asarray(stream, dtype=np.int64)
        if np.any(stream < 1) or stream.sum() < horizon:
            raise ValueError("Positive length streams must cover the decode horizon")
        end = np.cumsum(stream)
        keep = end <= horizon
        ends.append(end[keep])
        completed.append(stream[keep])
    ends = np.concatenate(ends)
    completed = np.concatenate(completed)
    order = np.argsort(ends, kind="stable")
    ends, completed = ends[order], completed[order]
    steps, starts, counts = np.unique(ends, return_index=True, return_counts=True)
    finished_length = (
        np.add.reduceat(completed, starts)
        if len(steps)
        else np.array([], dtype=np.int64)
    )
    if not len(steps) or steps[-1] < horizon:
        steps = np.append(steps, horizon)
        counts = np.append(counts, 0)
        finished_length = np.append(finished_length, 0)
    previous_step = np.r_[0, steps[:-1]]
    finished_before = np.r_[0, np.cumsum(finished_length)[:-1]]
    span = steps - previous_step
    # At step j+1, each generator has age j+1 minus all its past lengths.
    first_age_sum = batch * (previous_step + 1) - finished_before
    cache_sum = span * (batch * prompt + first_age_sum) + batch * span * (span - 1) // 2
    elapsed = np.cumsum(span * constant + cache_sum * kv_seconds)
    age_at_end = first_age_sum + batch * (span - 1)
    peak_positions = int(batch * prompt + np.max(age_at_end))
    return dict(
        times=np.repeat(elapsed, counts),
        lengths=completed,
        end_time=float(elapsed[-1]),
        peak_positions=peak_positions,
    )


def batch_samples(traces, n_sequences, warmup, batches):
    """Pool asynchronous replicas; select consecutive FIFO completion batches."""
    times = np.concatenate([trace["times"] for trace in traces])
    lengths = np.concatenate([trace["lengths"] for trace in traces])
    order = np.argsort(times, kind="stable")
    times, lengths = times[order], lengths[order]
    # Every replica must have been simulated through the observation end.
    common_end = min(trace["end_time"] for trace in traces)
    needed = (warmup + batches) * n_sequences
    if np.searchsorted(times, common_end, side="right") < needed:
        raise ValueError("Decode horizon does not cover all requested batches")
    boundaries = np.arange(warmup, warmup + batches + 1) * n_sequences - 1
    intervals = np.diff(times[boundaries])
    selected = lengths[warmup * n_sequences : needed].reshape(batches, n_sequences)
    return intervals, selected


def trainer_times(lengths, model, hardware, layout):
    """Same full-attention FLOP and TP/SP model as the companion solver."""
    ell = lengths.astype(float) + layout["prompt_tokens"]
    sums = ell.sum(axis=1)
    # Include the causal diagonal, as allocation._attention does.
    pairs = model["L"] * (ell * (ell + 1)).sum(axis=1) / 2
    return (
        _training_work(
            model,
            dict(mean_length=sums, attention_pairs=pairs),
            hardware["C"],
            hardware["W_ici"],
            layout["trainer_tp"],
            layout["epochs"],
        )
        / layout["trainer_gpus"]
    )


def simulate(mean_response, seed, batches, warmup=128, layout=LAYOUT):
    model, hw = MODELS[layout["model"]], HARDWARE[layout["hardware"]]
    if model["global_every"] != 1:
        raise ValueError("This preview integrates full-attention cache only")
    maximum = 4 * mean_response
    prompt, batch, tp = (
        layout["prompt_tokens"],
        layout["batch_per_replica"],
        layout["sampler_tp"],
    )
    if maximum + prompt > model["max_context"]:
        raise ValueError("Four-times-mean response cap exceeds the context limit")
    law = response_law("geometric", float(mean_response), maximum)
    p = dict(law)[1]  # Capping above one leaves P(T=1) unchanged.
    stats = _statistics(
        tuple(sorted(model.items())),
        "geometric",
        float(mean_response),
        maximum,
        prompt,
        1.0,
    )
    parameters = _model_parameters(model)
    kappa = 4 * model["L"] * model["NKV"] * model["d"]
    constant = (
        max(2 * batch * parameters / hw["C"], 2 * parameters / hw["W"])
        + 8 * model["L"] * batch * model["D"] * (tp - 1) / hw["W_ici"]
    ) / tp
    kv_seconds = kappa / (tp * hw["W"])
    available = layout["memory_fraction"] * hw["HBM"]
    sampler_memory = (2 * parameters + batch * kappa * (prompt + maximum)) / tp
    trainer_memory = (
        16 * parameters / layout["trainer_gpus"]
        + _activation_bytes(model, prompt + maximum) / layout["trainer_tp"]
    )
    if max(sampler_memory, trainer_memory) > available:
        raise ValueError("Fixed layout exceeds its memory budget at the response cap")

    # Padding covers random completion counts and different replica wall clocks.
    horizon = math.ceil(
        (warmup + batches + max(256, batches // 8))
        * layout["n_sequences"]
        * stats["mean_response"]
        / (layout["replicas"] * batch)
    )
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(layout["replicas"]):
        streams = []
        for _ in range(batch):
            expected = horizon / stats["mean_response"]
            size = math.ceil(expected + 10 * math.sqrt(expected) + 32)
            stream = np.minimum(rng.geometric(p, size=size), maximum)
            while stream.sum() < horizon:
                stream = np.r_[stream, np.minimum(rng.geometric(p, size=256), maximum)]
            streams.append(stream)
        traces.append(decode_trace(streams, horizon, prompt, constant, kv_seconds))
    sampler, selected = batch_samples(traces, layout["n_sequences"], warmup, batches)
    trainer = trainer_times(selected, model, hw, layout)
    step_mean = constant + batch * stats["kv_mean"] / (tp * hw["W"])
    predicted_sampler = (
        layout["n_sequences"]
        * stats["mean_response"]
        * step_mean
        / (layout["replicas"] * batch)
    )
    predicted_trainer = (
        layout["n_sequences"]
        / layout["trainer_gpus"]
        * _training_work(
            model, stats, hw["C"], hw["W_ici"], layout["trainer_tp"], layout["epochs"]
        )
    )
    return dict(
        sampler=sampler,
        trainer=trainer,
        predicted_sampler=predicted_sampler,
        predicted_trainer=predicted_trainer,
        sampler_memory_bytes=sampler_memory,
        trainer_memory_bytes=trainer_memory,
        observed_sampler_peak_bytes=max(
            (2 * parameters + kappa * t["peak_positions"]) / tp for t in traces
        ),
    )


def generate(out, batches, runs, warmup, seed):
    out.mkdir(parents=True, exist_ok=True)
    rows, raw, run_rows = [], {}, []
    for ell in LENGTHS:
        samples = {stage: [] for stage in ("sampler", "trainer")}
        results = []
        for run in range(runs):
            run_seed = seed + run  # Common random numbers across length settings.
            result = simulate(ell - LAYOUT["prompt_tokens"], run_seed, batches, warmup)
            results.append(result)
            for stage in samples:
                values = result[stage]
                samples[stage].append(values)
                run_rows.append(
                    dict(
                        mean_total_length=ell,
                        stage=stage,
                        seed=run_seed,
                        mean=float(np.mean(values)),
                        p99=float(np.quantile(values, 0.99)),
                    )
                )
        row = dict(
            mean_total_length=ell,
            mean_response=ell - LAYOUT["prompt_tokens"],
            max_response=4 * (ell - LAYOUT["prompt_tokens"]),
            samples=runs * batches,
            sampler_memory_bytes=result["sampler_memory_bytes"],
            trainer_memory_bytes=result["trainer_memory_bytes"],
            observed_sampler_peak_bytes=max(
                r["observed_sampler_peak_bytes"] for r in results
            ),
        )
        for stage, chunks in samples.items():
            values = np.concatenate(chunks)
            raw[f"{stage}_{ell}"] = values
            mean, p99 = float(np.mean(values)), float(np.quantile(values, 0.99))
            predicted = result["predicted_" + stage]
            row.update(
                {
                    stage + "_mean_seconds": mean,
                    stage + "_p99_seconds": p99,
                    stage + "_predicted_mean_seconds": predicted,
                    stage + "_mean_relative_error": mean / predicted - 1,
                    stage + "_p99_excess": p99 / mean - 1,
                }
            )
            ratios = [
                float(np.quantile(chunk, 0.99) / np.mean(chunk) - 1) for chunk in chunks
            ]
            row[stage + "_run_excess_min"] = min(ratios)
            row[stage + "_run_excess_max"] = max(ratios)
        rows.append(row)
        print(
            f"{ell:5d} tokens: sampler {row['sampler_mean_seconds']:.3f}s / "
            f"{row['sampler_p99_seconds']:.3f}s p99; trainer "
            f"{row['trainer_mean_seconds']:.3f}s / {row['trainer_p99_seconds']:.3f}s p99",
            flush=True,
        )
    np.savez_compressed(out / "batch-times.npz", **raw)
    data = dict(
        layout=LAYOUT,
        batches_per_run=batches,
        runs=runs,
        warmup_batches=warmup,
        seed=seed,
        numpy_version=np.__version__,
        quantile_method="linear",
        rows=rows,
        run_statistics=run_rows,
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
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 12,
            "legend.frameon": False,
        }
    )
    colors = (
        matplotlib.colormaps["cividis"](0.17),
        matplotlib.colormaps["magma"](0.57),
    )
    rows = data["rows"]
    x = [row["mean_total_length"] for row in rows]

    def frame(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS, labels=["1k", "2k", "4k", "8k", "16k", "32k"])
        ax.set_xlabel(r"Mean total sequence length $\mathbb{E}[\ell]$ (tokens)")
        ax.grid(axis="y", color=".9", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, stage, color, title in zip(
        axes, ("sampler", "trainer"), colors, ("Sampler: 24 GPUs", "Trainer: 8 GPUs")
    ):
        ax.plot(
            x,
            [r[stage + "_mean_seconds"] for r in rows],
            color=color,
            marker="o",
            lw=2,
            ms=4,
        )
        ax.plot(
            x,
            [r[stage + "_p99_seconds"] for r in rows],
            color=color,
            marker="o",
            ls="--",
            lw=2,
            ms=4,
        )
        ax.set_yscale("log")
        ax.set_title(title, loc="center")
        frame(ax)
    axes[0].set_ylabel("Batch time (seconds)")
    fig.legend(
        handles=[
            Line2D([], [], color=".25", lw=2, label="Mean"),
            Line2D([], [], color=".25", lw=2, ls="--", label="99th percentile"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
    )
    fig.suptitle("Batch time by mean sequence length", x=0.5, ha="center", fontsize=16)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.16, top=0.73, wspace=0.1)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"batch-times.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for stage, color in zip(("sampler", "trainer"), colors):
        ax.plot(
            x,
            [r[stage + "_p99_excess"] for r in rows],
            color=color,
            marker="o",
            lw=2,
            label=stage.capitalize(),
        )
        ax.fill_between(
            x,
            [r[stage + "_run_excess_min"] for r in rows],
            [r[stage + "_run_excess_max"] for r in rows],
            color=color,
            alpha=0.10,
        )
    frame(ax)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("99th percentile / mean − 1")
    ax.set_ylim(bottom=0)
    ax.set_title("99th-percentile batch time above the mean", loc="center", pad=14)
    ax.legend(loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"batch-time-spread.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def write_notes(data, out):
    rows = data["rows"]
    last = rows[-1]
    errors = [
        abs(r[stage + "_mean_relative_error"])
        for r in rows
        for stage in ("sampler", "trainer")
    ]
    table = [
        "| Mean total tokens | Sampler mean / p99 (s) | Trainer mean / p99 (s) |",
        "|---:|---:|---:|",
    ]
    table += [
        f"| {r['mean_total_length']:,} | {r['sampler_mean_seconds']:.2f} / {r['sampler_p99_seconds']:.2f} "
        f"| {r['trainer_mean_seconds']:.2f} / {r['trainer_p99_seconds']:.2f} |"
        for r in rows
    ]
    (out / "README.md").write_text(
        f"""# Batch timing — plots for review

These plots are separate from `main.md`.

![Mean and 99th-percentile batch times](batch-times.png)

*Llama 3.1 8B on 32 H100s. Fixed layout: three sampler replicas, each with TP8 and 32 generators (24 GPUs); the trainer uses eight FSDP ranks with TP1. Each fresh batch has 1,024 sequences, prompts have 512 tokens, and training uses one epoch with full activation recompute. Geometric responses are capped at four times their requested mean; the hazard is adjusted so the capped mean equals that request. Solid lines show the mean; dashed lines show the 99th percentile. The same sampled completion batches feed the trainer cost model. No weight sync or prefill is included.*

![99th-percentile excess above the mean](batch-time-spread.png)

*The vertical axis is the extra time at the 99th percentile, relative to the mean. For example, 10% means p99 is 1.10 times the mean. The faint bands show the range across {data['runs']} independent runs; they are not confidence intervals.*

{chr(10).join(table)}

At 32k mean total length, the sampler's 99th percentile is {last['sampler_p99_excess']:.1%} above its mean; the trainer's is {last['trainer_p99_excess']:.1%} above its mean. The gap increases in this example as KV reads and training attention account for more of the work. These values depend on the fixed layout, batch size and length law.

The sampler time is the interval between successive batches of 1,024 completions after warm-up. Replicas have independent decode clocks. A generator immediately starts its next response after finishing. Completions are pooled in time order, and whole sequences enter consecutive FIFO batches. Surplus completions at one event go into the next batch. Ties use a fixed order.

The sampler simulation sums the article's affine step-time formula using the actual aggregate cache. Between completion events, all cache ages increase by one per decode step, so the step times form an arithmetic progression. The code sums that progression exactly without visiting each token. It keeps the article's convention of counting the current token in cache occupancy.

Trainer times use each completed batch's token count and attention-pair count. As in the companion solver, causal attention includes the diagonal; Section 2's leading-order squared-length term omits that small correction. The work is divided evenly across trainer GPUs. No rank imbalance, padding, variable kernel efficiency or collective latency is added.

Each length setting uses {data['runs']:,} independent runs of {data['batches_per_run']:,} batches after {data['warmup_batches']:,} warm-up batches per run: {data['runs'] * data['batches_per_run']:,} recorded batches. Seeds start at {data['seed']}; a run reuses its seed across the length settings. The largest difference between a simulated mean and its analytical prediction is {max(errors):.2%}. This checks the mean calculation; it does not validate the model against GPU measurements.

The fixed layout fits even if every sampler response reaches the cap. Trainer memory also fits at the capped length, with one sequence per micro-batch. Both pools use at most 90% of H100 memory under the article's memory model. The JSON records worst-case budgets and the largest observed sampler occupancy.

This is an unpaced sampler and a separate trainer cost calculation. It does not simulate their queue, trainer waiting, weight updates or staleness. It also does not optimize the GPU split. A p99 batch interval alone does not determine how often the trainer waits. Its distribution depends on batch size, generator count and the response-length law. The curves include only timing variation caused by random lengths; they are not measured GPU percentiles.

Reproduce from the repository root (NumPy and Matplotlib required):

```sh
python -m utils.make_batch_timing_preview
```

`summary.csv` and `results.json` contain the estimates and settings. `batch-times.npz` contains all recorded batch times. The generator does not update the article or its figures.
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--batches-per-run", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if min(args.batches_per_run, args.runs, args.warmup) < 1:
        parser.error("Run, batch and warm-up counts must be positive")
    data = generate(
        args.output_dir, args.batches_per_run, args.runs, args.warmup, args.seed
    )
    plot(data, args.output_dir)
    write_notes(data, args.output_dir)
    print(f"Review figures and data: {args.output_dir}")
