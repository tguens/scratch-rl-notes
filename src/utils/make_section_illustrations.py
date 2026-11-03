"""Cost breakdown and staleness illustrations for the article.

Run from the repository root: python -m utils.make_section_illustrations
Requires Matplotlib and Pillow. Outputs go to figs/main/ by default.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from allocation import (
    HARDWARE,
    MODELS,
    _activation_bytes,
    _model_parameters,
    _training_work,
)
from utils.make_generator_staleness import render as token_staleness

OUT = Path(__file__).resolve().parents[2] / "figs/main"
LENGTHS = (1024, 2048, 3072, 4096, 8192, 16384, 32768)
INK = "#242424"


def save(fig, out, name):
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def stage_costs(out):
    model, hw = MODELS["Llama-3.1-8B"], HARDWARE["H100"]
    parameters = _model_parameters(model)
    kappa = 4 * model["L"] * model["NKV"] * model["d"]
    query_width = model["NQ"] * model["d"]
    prompt, sampler_tp, trainer_gpus = 512, 2, 2
    usable = 0.9 * hw["HBM"]
    rows = []
    for ell in LENGTHS:
        response = ell - prompt
        mean_cache = prompt + (response + 1) / 2
        batch = min(
            math.ceil(hw["C"] / hw["W"]),
            math.floor((sampler_tp * usable - 2 * parameters) / (kappa * ell)),
        )
        assert batch >= 1
        row = dict(
            total_length=ell,
            response_length=response,
            batch_per_replica=batch,
            sampler_linear=response
            * max(2 * parameters / hw["C"], 2 * parameters / (batch * hw["W"])),
            sampler_kv=response * kappa * mean_cache / hw["W"],
            sampler_tp=response
            * 8
            * model["L"]
            * model["D"]
            * (sampler_tp - 1)
            / hw["W_ici"],
            trainer_linear=8 * parameters * ell / hw["C"],
            trainer_attention=9 * model["L"] * query_width * ell**2 / hw["C"],
            sampler_memory_bytes=(2 * parameters + kappa * batch * ell) / sampler_tp,
            trainer_memory_bytes=16 * parameters / trainer_gpus
            + _activation_bytes(model, ell),
        )
        row["sampler_total"] = sum(
            row[k] for k in ("sampler_linear", "sampler_kv", "sampler_tp")
        )
        row["trainer_total"] = row["trainer_linear"] + row["trainer_attention"]
        assert max(row["sampler_memory_bytes"], row["trainer_memory_bytes"]) <= usable
        # Cross-check the component sum against the article's decode-step formula.
        step = (
            max(2 * batch * parameters / hw["C"], 2 * parameters / hw["W"])
            + kappa * batch * mean_cache / hw["W"]
            + 8 * model["L"] * batch * model["D"] * (sampler_tp - 1) / hw["W_ici"]
        ) / sampler_tp
        assert math.isclose(
            row["sampler_total"], sampler_tp * response * step / batch, rel_tol=1e-12
        )
        # The solver includes the causal diagonal; the article uses leading ell².
        exact = _training_work(
            model,
            dict(mean_length=ell, attention_pairs=model["L"] * ell * (ell + 1) / 2),
            hw["C"],
            hw["W_ici"],
            1,
            1,
        )
        diagonal = 9 * model["L"] * query_width * ell / hw["C"]
        assert math.isclose(row["trainer_total"], exact - diagonal, rel_tol=1e-12)
        for micro in (1, 2, 4):
            row[f"trainer_memory_bmu{micro}_bytes"] = (
                16 * parameters / trainer_gpus + micro * _activation_bytes(model, ell)
            )
        rows.append(row)

    colors = [matplotlib.colormaps["magma"](v) for v in (0.22, 0.56, 0.78)]
    training_colors = [matplotlib.colormaps["cividis"](v) for v in (0.14, 0.65)]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.1), sharex=True, sharey=True)
    x = [row["total_length"] for row in rows]
    definitions = (
        (
            "Sampler: TP2",
            "sampler",
            (
                ("sampler_linear", "Linear layers"),
                ("sampler_kv", "KV reads"),
                ("sampler_tp", "TP communication"),
            ),
            colors,
        ),
        (
            "Trainer: TP1 / DP2 (FSDP)",
            "trainer",
            (
                ("trainer_linear", "Linear layers"),
                ("trainer_attention", "Attention scores + values"),
            ),
            training_colors,
        ),
    )
    for ax, (title, stage, components, palette) in zip(axes, definitions):
        ax.plot(
            x,
            [r[stage + "_total"] for r in rows],
            color=INK,
            lw=2.8,
            marker="o",
            ms=4,
            label="Total",
            zorder=4,
        )
        for (key, label), color in zip(components, palette):
            ax.plot(x, [r[key] for r in rows], color=color, lw=1.9, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(
            [1024, 2048, 4096, 8192, 16384, 32768],
            labels=["1k", "2k", "4k", "8k", "16k", "32k"],
        )
        ax.set_xlabel(r"Total sequence length $\ell$ (tokens)")
        ax.set_title(title, loc="center", pad=12)
        ax.grid(axis="y", color=".9", lw=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("GPU-seconds per completed sequence")
    axes[0].set_ylim(7e-4, 65)
    for ell, offset in ((1024, (7, 19)), (4096, (-8, 23)), (32768, (-8, 18))):
        row = next(r for r in rows if r["total_length"] == ell)
        axes[0].annotate(
            f"B = {row['batch_per_replica']}",
            (ell, row["sampler_total"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            ha="left" if ell == 1024 else "right",
            color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=1.2),
            arrowprops=dict(arrowstyle="-", color=".6", lw=0.7),
        )
    inset = axes[1].inset_axes([0.59, 0.06, 0.38, 0.33])
    inset.axhspan(usable / 1e9, 115, color=".95")
    memory_colors = [matplotlib.colormaps["cividis"](v) for v in (0.12, 0.45, 0.78)]
    for micro, color in zip((1, 2, 4), memory_colors):
        inset.plot(
            x,
            [r[f"trainer_memory_bmu{micro}_bytes"] / 1e9 for r in rows],
            color=color,
            lw=1.3,
            label=rf"$b_\mu={micro}$",
        )
    inset.axhline(usable / 1e9, color=".4", lw=0.8, ls="--")
    inset.set_xscale("log", base=2)
    inset.set_xticks([1024, 8192, 32768], labels=["1k", "8k", "32k"])
    inset.set_ylim(52, 115)
    inset.set_yticks([60, 72, 100])
    inset.set_ylabel("GB/GPU", fontsize=7, labelpad=1)
    inset.set_title("Micro-batch memory", fontsize=8.5, pad=5)
    inset.tick_params(labelsize=7, length=2, pad=2)
    inset.legend(
        loc="upper left",
        fontsize=6.5,
        ncol=3,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.4,
        borderaxespad=0.25,
    )
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    fig.suptitle(
        "Sampling and training cost by sequence length", fontsize=16, x=0.5, ha="center"
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.83, wspace=0.12)
    save(fig, out, "stage-costs")
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    with (data_dir / "stage-costs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(
        model="Llama-3.1-8B",
        hardware="H100",
        prompt_tokens=prompt,
        usable_memory_bytes=usable,
        sampler_tp=sampler_tp,
        trainer_tp=1,
        trainer_fsdp=2,
        parameters=parameters,
        kv_bytes_per_token=kappa,
        rows=rows,
    )


def write_data(out, costs, staleness):
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "section-illustrations.json").write_text(
        json.dumps(dict(costs=costs, staleness=staleness), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    costs = stage_costs(args.output_dir)
    staleness = token_staleness(args.output_dir)
    write_data(args.output_dir, costs, staleness)
    print(f"Wrote two checked article illustrations to {args.output_dir}")
