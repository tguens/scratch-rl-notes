"""Staleness animation in the article's four-generator visual style."""

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

COLORS = ("#5d9ed6", "#63b97c", "#f1c64f", "#e8756c")
LENGTHS = ((9,), (3, 3, 3), (1, 2, 1, 2, 3), (3, 3, 3))
LEFT_X, CELL, TILE = 12.0, 6.0, 4.55
ROWS = (12.0, 18.2, 24.4, 30.6)
QUEUE_X, QUEUE_CELL, QUEUE_TILE = 91.0, 4.35, 3.5
QUEUE_ROWS = (13.0, 19.0, 25.0, 31.0)


def story():
    """Exact FIFO hand-scripted toy; four completions per trainer cycle."""
    state = dict(
        buffers=[[] for _ in range(4)],
        queue=[],
        used=None,
        trainer_step=1,
        sampler_version=0,
        tick=0,
        pause=False,
        growth=1.0,
        growing=[],
        eos=[],
        moving=[],
        flight=0.0,
        fade=1.0,
        show_staleness=False,
        initial=True,
    )
    frames, consumed = [], []
    indices = [0] * 4

    def hold(count=1):
        frames.extend(copy.deepcopy(state) for _ in range(count))

    hold(6)
    for tick in range(1, 10):
        version = (tick - 1) // 3
        assert state["sampler_version"] == version
        state.update(tick=tick, initial=False, growing=list(range(4)), eos=[])
        finished = []
        for g in range(4):
            state["buffers"][g].append(version)
            if len(state["buffers"][g]) == LENGTHS[g][indices[g]]:
                finished.append(dict(generator=g, versions=state["buffers"][g][:]))
                state["eos"].append(g)
                indices[g] += 1
        for growth in (0.3, 0.65, 1.0):
            state["growth"] = growth
            hold()
        state.update(growing=[], growth=1.0)
        if finished:
            assert len(state["queue"]) + len(finished) <= 4
            state["moving"] = [
                dict(sequence=seq, target=len(state["queue"]) + i)
                for i, seq in enumerate(finished)
            ]
            for seq in finished:
                state["buffers"][seq["generator"]] = []
            state["eos"] = []
            for progress in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                state["flight"] = progress
                hold()
            state["queue"].extend(finished)
            state.update(moving=[], flight=0.0)
        hold(2)
        if tick % 3 == 0:
            assert len(state["queue"]) == 4
            hold(4)
            state["trainer_step"] += 1
            state["used"], state["queue"] = state["queue"], []
            consumed.append(
                dict(step=state["trainer_step"], sequences=copy.deepcopy(state["used"]))
            )
            state["pause"] = True
            if tick < 9:
                for fade in (1.0, 0.55, 0.15):
                    state["fade"] = fade
                    hold()
                state.update(used=None, fade=1.0)
            hold(5)
            state.update(sampler_version=state["trainer_step"] - 1, pause=False)
            hold(2)
    state["show_staleness"] = True
    hold(38)
    final = copy.deepcopy(state)
    assert final["trainer_step"] == 4 and final["sampler_version"] == 3
    assert final["used"][0]["versions"] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert all(seq["versions"] == [2, 2, 2] for seq in final["used"][1:])
    assert all(len(batch["sequences"]) == 4 for batch in consumed)
    return frames, final, consumed


def darker(color):
    return tuple(int(color[j : j + 2], 16) / 255 * 0.58 for j in (1, 3, 5))


def draw(state, fig):
    fig.clear()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 140)
    ax.set_ylim(44, 0)
    ax.axis("off")
    ax.text(
        70,
        2.3,
        "Weight updates during generation",
        ha="center",
        va="center",
        fontsize=19,
    )
    current = state["trainer_step"] - 1
    ax.text(
        5,
        6.6,
        f"Trainer step {state['trainer_step']} · weights {current}",
        fontsize=12,
        color=".25",
    )
    ax.text(
        135,
        6.6,
        f"Sampler weights {state['sampler_version']}",
        ha="right",
        fontsize=12,
        color=".25",
    )
    if state["pause"]:
        ax.text(
            LEFT_X, 10.1, f"Sync → v{current}: sampler paused", fontsize=12, color=".3"
        )
    elif state["initial"]:
        ax.text(
            LEFT_X,
            10.1,
            "Step 1 uses a batch prepared earlier",
            fontsize=10,
            color=".45",
        )

    def tile(
        x, y, width, version, generator, eos=False, alpha=1.0, scale=1.0, small=False
    ):
        face = darker(COLORS[generator]) if eos else COLORS[generator]
        margin = width * (1 - scale) / 2
        ax.add_patch(
            Rectangle(
                (x + margin, y + margin),
                width * scale,
                width * scale,
                facecolor=face,
                edgecolor="none",
                alpha=alpha,
            )
        )
        if scale < 0.6:
            return
        ax.text(
            x + width / 2,
            y + width * (0.40 if eos else 0.52),
            f"v{version}",
            ha="center",
            va="center",
            fontsize=10 if small else 13,
            color="white" if eos or generator in (0, 3) else "#253c29",
            alpha=alpha,
        )
        if eos:
            ax.text(
                x + width / 2,
                y + width * 0.81,
                "EOS",
                ha="center",
                va="center",
                fontsize=5 if small else 6,
                color="white",
                alpha=alpha,
            )

    for g, y in enumerate(ROWS):
        ax.text(
            7.5,
            y + 2.5,
            rf"$g_{{{g+1}}}$",
            ha="center",
            va="center",
            fontsize=21,
            color=COLORS[g],
        )
        for j in range(9):
            ax.add_patch(
                Rectangle(
                    (LEFT_X + j * CELL, y),
                    CELL,
                    5.0,
                    facecolor="white",
                    edgecolor=".84",
                    lw=0.7,
                )
            )
        versions = state["buffers"][g]
        for j, version in enumerate(versions):
            growing = g in state["growing"] and j == len(versions) - 1
            tile(
                LEFT_X + j * CELL + 0.72,
                y + 0.23,
                TILE,
                version,
                g,
                eos=g in state["eos"] and j == len(versions) - 1,
                scale=state["growth"] if growing else 1.0,
            )
    ax.text(LEFT_X, 39.9, f"Decode step {state['tick']}", fontsize=11, color=".3")
    ax.text(LEFT_X + 23, 39.9, "v = weight version", fontsize=11, color=".3")

    used = state["used"]
    title = (
        f"Batch {state['trainer_step']} in training"
        if used is not None
        else "Trainer buffer"
    )
    ax.text(88, 10.0, title, fontsize=16)
    landed = len(state["moving"]) if state["flight"] >= 1 else 0
    count = (
        f"{len(used)} sequences"
        if used is not None
        else f"{len(state['queue']) + landed} / 4 sequences"
    )
    ax.text(135, 10.0, count, fontsize=11, color=".35", ha="right")
    ax.add_patch(
        FancyBboxPatch(
            (88, 11.4),
            47,
            26.5,
            boxstyle="round,pad=0,rounding_size=1.1",
            facecolor="#f1f1f1",
            edgecolor="none",
        )
    )
    sequences = used if used is not None else state["queue"]
    for row, seq in enumerate(sequences):
        for j, version in enumerate(seq["versions"]):
            x, y = QUEUE_X + j * QUEUE_CELL, QUEUE_ROWS[row]
            tile(
                x,
                y,
                QUEUE_TILE,
                version,
                seq["generator"],
                eos=j == len(seq["versions"]) - 1,
                alpha=state["fade"],
                small=True,
            )
            if state["show_staleness"]:
                ax.text(
                    x + QUEUE_TILE / 2,
                    y + 4.8,
                    str(current - version),
                    fontsize=11,
                    ha="center",
                    va="center",
                    color=".25",
                )
    for moving in state["moving"]:
        seq, target, p = moving["sequence"], moving["target"], state["flight"]
        g = seq["generator"]
        for j, version in enumerate(seq["versions"]):
            x0, y0 = LEFT_X + j * CELL + 0.72, ROWS[g] + 0.23
            x1, y1 = QUEUE_X + j * QUEUE_CELL, QUEUE_ROWS[target]
            tile(
                x0 + (x1 - x0) * p,
                y0 + (y1 - y0) * p,
                TILE + (QUEUE_TILE - TILE) * p,
                version,
                g,
                eos=j == len(seq["versions"]) - 1,
                small=p > 0.5,
            )
    if state["show_staleness"]:
        ax.text(
            88,
            40.1,
            f"Numbers below tokens: staleness = {current} − v",
            fontsize=11,
            color=".25",
        )


def render(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames, final, batches = story()
    fig = plt.figure(figsize=(14, 4.4), dpi=120)
    draw(final, fig)
    fig.savefig(out / "token-staleness.png", dpi=160, facecolor="white")
    fig.savefig(out / "token-staleness.svg", facecolor="white")
    fig.canvas.draw()
    poster = Image.frombytes(
        "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
    ).convert("RGB")
    palette = poster.quantize(colors=256)
    images = []
    for state in frames:
        draw(state, fig)
        fig.canvas.draw()
        image = Image.frombytes(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        ).convert("RGB")
        images.append(image.quantize(palette=palette, dither=Image.Dither.NONE))
    images[0].save(
        out / "token-staleness.gif",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        disposal=2,
        optimize=False,
    )
    plt.close(fig)
    result = dict(
        generators=4,
        sequences_per_batch=4,
        ticks_per_cycle=3,
        response_lengths=LENGTHS,
        prepared_first_batch=True,
        consumed=batches,
        final_staleness=[[3 - v for v in seq["versions"]] for seq in final["used"]],
        schematic=True,
        frames=len(frames),
        milliseconds_per_frame=100,
    )
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "generator-staleness-data.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        f"Rendered {len(frames)} frames; checked FIFO batches and token versions.",
        flush=True,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
        }
    )
    render(args.output_dir)
