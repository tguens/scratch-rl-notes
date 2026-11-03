"""Two allocation master plots and a supporting throughput comparison.

Run from the repository root: python -m utils.make_main_figures
The allocator uses only the standard library; plotting requires matplotlib.
Captions are kept in the article, outside the plotted images.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from allocation import HARDWARE, MODELS, allocate

OUT = Path(__file__).resolve().parents[2] / "figs" / "main"
NAMES = {
    "article-master-allocation": "sampler-allocation",
    "article-master-allocation-laws": "sampler-allocation-laws",
    "article-master-length-laws": "response-length-throughput",
}
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 40960, 65536)
BUDGETS = (16, 64)
BASE = dict(
    prompt_tokens=512,
    n_sequences=1024,
    epochs=1,
    memory_fraction=0.9,
    mean_staleness_limit=8,
    weight_sync=True,
    sampler_efficiency=1.0,
    trainer_efficiency=1.0,
    staggered_memory="fluid",
)
LABELS = ("Llama 3.1 8B", "Qwen3 8B", "OLMo 3 7B", "Gemma 3 12B")


def solve(n_gpus, model, hardware, experiment, **kwargs):
    r = allocate(n_gpus, model, hardware, **(BASE | kwargs))
    assert r["feasible"], (n_gpus, model, hardware, kwargs)
    assert r["sampler_gpus"] + r["trainer_gpus"] + r["unused_gpus"] == n_gpus
    assert r["sampler_memory_bytes"] <= r["usable_memory_bytes"] * (1 + 1e-10)
    assert r["trainer_memory_bytes"] <= r["usable_memory_bytes"] * (1 + 1e-10)
    assert (
        r["estimated_mean_first_token_staleness"]
        <= r["inputs"]["mean_staleness_limit"] + 1e-10
    )
    cycle = max(r["sampler_seconds"] + r["sync_seconds"], r["trainer_seconds"])
    assert abs(r["sequences_per_second"] * cycle / BASE["n_sequences"] - 1) < 1e-10
    r["experiment"] = experiment
    return r


def frame(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=".90", lw=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{NAMES[name]}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def allocation_plot():
    colors = [matplotlib.colormaps["magma"](v) for v in (0.13, 0.37, 0.61, 0.79)]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), sharex=True, sharey=True)
    rows = []
    for ax, hardware in zip(axes, HARDWARE):
        for n_gpus, style in zip(BUDGETS, ("-", "--")):
            for model, label, color in zip(MODELS, LABELS, colors):
                lengths = [
                    ell for ell in LENGTHS if ell <= MODELS[model]["max_context"]
                ]
                series = [
                    solve(
                        n_gpus,
                        model,
                        hardware,
                        "allocation",
                        arrival="staggered",
                        mean_response=ell - 512,
                        max_response=ell - 512,
                    )
                    for ell in lengths
                ]
                rows.extend(series)
                ax.plot(
                    lengths,
                    [r["sampler_gpus"] / n_gpus for r in series],
                    color=color,
                    ls=style,
                    lw=1.9,
                    marker="o" if n_gpus == BUDGETS[0] else "s",
                    ms=2.7,
                )
        ax.set_title(
            hardware if hardware != "B200" else "B200 (Blackwell)", loc="center"
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 4096, 16384, 65536], labels=["1k", "4k", "16k", "64k"])
        ax.set_xlabel(r"Total sequence length $\ell$ (tokens)")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        frame(ax)
    axes[0].set_ylabel("Sampler GPUs / total GPUs")
    handles = [
        Line2D([], [], color=c, lw=2, label=label) for c, label in zip(colors, LABELS)
    ]
    handles += [
        Line2D(
            [], [], color=".25", lw=1.9, ls=style, marker=marker, ms=3, label=f"N = {n}"
        )
        for n, style, marker in zip(BUDGETS, ("-", "--"), ("o", "s"))
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=6,
        fontsize=9,
    )
    fig.suptitle(
        "Sampler GPU share by sequence length and GPU budget",
        x=0.5,
        ha="center",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.14, top=0.77, wspace=0.10)
    save(fig, "article-master-allocation")
    return rows


def length_law_plot():
    means = (1024, 2048, 4096, 8192, 16384, 32768)
    laws = (
        ("staggered", "Equal, staggered (2)"),
        ("synchronized", "Equal, synchronized (1)"),
        ("geometric", "Geometric (3)"),
        ("lognormal", "Log-normal (4)"),
    )
    colors = [matplotlib.colormaps["cividis"](v) for v in (0.03, 0.34, 0.60, 0.84)]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    rows, baseline = [], None
    for (law, label), color, marker in zip(laws, colors, ("o", "s", "^", "D")):
        series = [
            solve(
                32,
                "Llama-3.1-8B",
                "H100",
                "length_laws",
                arrival=law,
                mean_response=ell - 512,
                max_response=4 * (ell - 512),
            )
            for ell in means
        ]
        if baseline is None:
            baseline = [r["sequences_per_second"] for r in series]
        for r, ref in zip(series, baseline):
            r["relative_rate"] = r["sequences_per_second"] / ref
        rows.extend(series)
        ax.plot(
            means,
            [r["relative_rate"] for r in series],
            color=color,
            marker=marker,
            ms=4,
            lw=2,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(means, labels=["1k", "2k", "4k", "8k", "16k", "32k"])
    ax.set_xlabel(r"Mean total sequence length $\mathbb{E}[\ell]$ (tokens)")
    ax.set_ylabel("Fresh-sequence rate / equal-staggered rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="lower left", fontsize=9)
    frame(ax)
    fig.suptitle(
        "Fresh-sequence throughput by response-length model",
        x=0.5,
        ha="center",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.14, top=0.85)
    save(fig, "article-master-length-laws")
    return rows


def allocation_laws_plot():
    # A common range keeps the response cap at 4*mean for every checkpoint.
    # Qwen's configured 40,960-token context would truncate that cap above it.
    means = (1024, 2048, 4096, 8192, 10240)
    models = ("Llama-3.1-8B", "Qwen3-8B", "OLMo-3-7B")
    laws = (
        ("staggered", "Staggered (2)", "-", "o"),
        ("geometric", "Geometric (3)", "--", "s"),
        ("lognormal", "Log-normal (4)", ":", "^"),
    )
    colors = [matplotlib.colormaps["magma"](v) for v in (0.13, 0.37, 0.61)]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), sharex=True, sharey=True)
    rows = []
    for ax, hardware in zip(axes, HARDWARE):
        for model, color in zip(models, colors):
            for law, label, style, marker in laws:
                series = [
                    solve(
                        64,
                        model,
                        hardware,
                        "allocation_laws",
                        arrival=law,
                        mean_response=ell - 512,
                        max_response=4 * (ell - 512),
                    )
                    for ell in means
                ]
                rows.extend(series)
                ax.plot(
                    means,
                    [r["sampler_gpus"] / 64 for r in series],
                    color=color,
                    ls=style,
                    lw=1.9,
                    marker=marker,
                    ms=3,
                )
        ax.set_title(
            hardware if hardware != "B200" else "B200 (Blackwell)", loc="center"
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(means, labels=["1k", "2k", "4k", "8k", "10k"])
        ax.set_xlabel(r"Mean total sequence length $\mathbb{E}[\ell]$ (tokens)")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        frame(ax)
    axes[0].set_ylabel("Sampler GPUs / total GPUs")
    handles = [
        Line2D([], [], color=c, lw=2, label=label)
        for c, label in zip(colors, LABELS[:3])
    ]
    handles += [
        Line2D([], [], color=".25", lw=1.9, ls=style, marker=marker, ms=3, label=label)
        for law, label, style, marker in laws
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=6,
        fontsize=9,
    )
    fig.suptitle(
        "Sampler GPU share by response-length model", x=0.5, ha="center", fontsize=15
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.14, top=0.77, wspace=0.10)
    save(fig, "article-master-allocation-laws")
    return rows


def eta_controls():
    return [
        solve(
            64,
            "Llama-3.1-8B",
            "H100",
            "eta_sensitivity",
            arrival="staggered",
            mean_response=8192 - 512,
            max_response=8192 - 512,
            mean_staleness_limit=eta,
        )
        for eta in (2, 8)
    ]


def write_data(rows):
    clean = [
        {k: v for k, v in r.items() if k not in ("placement", "assumptions")}
        for r in rows
    ]
    data_dir = OUT / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "article-master-results.json").write_text(
        json.dumps(clean, indent=2) + "\n"
    )
    fields = (
        "experiment",
        "model",
        "hardware",
        "n_gpus",
        "sampler_gpus",
        "trainer_gpus",
        "sampler_tp",
        "trainer_tp",
        "batch_per_replica",
        "sequences_per_second",
        "trainer_wait_fraction",
        "trainer_at_memory_floor",
        "relative_rate",
    )
    with (data_dir / "article-master-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields + ("arrival", "mean_length", "eta")
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **{k: r.get(k) for k in fields},
                    "arrival": r["inputs"]["arrival"],
                    "mean_length": r["statistics"]["mean_length"],
                    "eta": r["inputs"]["mean_staleness_limit"],
                }
            )
    summary = {
        "32k_allocations": [
            {
                k: r[k]
                for k in (
                    "model",
                    "hardware",
                    "n_gpus",
                    "sampler_gpus",
                    "trainer_gpus",
                    "trainer_wait_fraction",
                )
            }
            for r in rows
            if r["experiment"] == "allocation"
            and abs(r["statistics"]["mean_length"] - 32768) < 1e-6
        ],
        "32k_law_rates": [
            {
                "law": r["inputs"]["arrival"],
                "relative_rate": r["relative_rate"],
                "sequences_per_second": r["sequences_per_second"],
            }
            for r in rows
            if r["experiment"] == "length_laws"
            and abs(r["statistics"]["mean_length"] - 32768) < 1e-6
        ],
    }
    (data_dir / "article-master-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Figure directory (default: figs/main). Raw results go in data/.",
    )
    OUT = parser.parse_args().output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )
    data = (
        allocation_plot() + length_law_plot() + allocation_laws_plot() + eta_controls()
    )
    write_data(data)
    print(
        f"Wrote {len(data)} checked allocations, two master plots and the supporting throughput plot."
    )
