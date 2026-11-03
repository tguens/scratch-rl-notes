# scratch-rl-notes

_Context: If you're building your own async RL system, you may have to ask yourself: how many GPUs do I need for my sampler / trainer? This is an attempt at answering with a simple theoretical model (written over the weekend)._

- Article: https://tguens.github.io/scratch-rl-notes/

A simple cost model for splitting GPUs between an async RL sampler and trainer.
- [(Markdown) Feeding the trainer](docs/main.md)
- [(Markdown) Appendix: derivations, assumptions and extensions](docs/_appendix.md)
- Companion code: [integer solver](src/allocation.py), [continuous solver](src/continuous_allocation.py)

## Setup

Use Python 3.9 or later. From the repository root, install the project and its plotting and development dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

The solvers use only the standard library; figures also use NumPy, Matplotlib and Pillow. No GPU is required. These are cost-model predictions, not measured GPU throughput.

After setup, these imports work from the repository root:

```python
from allocation import allocate
from continuous_allocation import continuous

integer = allocate(32, model="Llama-3.1-8B", hardware="H100")
relaxed = continuous(32, model="Llama-3.1-8B", hardware="H100")
print(integer)
print(relaxed)
```

Both solvers default to `staggered_memory="fluid"`, which budgets mean cache for
staggered responses. See the appendix for the assumptions and limitations.

## Regenerate the article figures

Run both commands from the repository root:

```sh
uv run python -m utils.make_main_figures
uv run python -m utils.make_section_illustrations
```

| Command | Figures in `figs/main/` |
|---|---|
| `utils.make_main_figures` | `sampler-allocation`, `sampler-allocation-laws`, `response-length-throughput` (PNG and SVG) |
| `utils.make_section_illustrations` | `stage-costs` (PNG and SVG), `token-staleness` (GIF, PNG and SVG) |

Numerical settings and results are saved in `figs/main/data/`.
The plots use Times New Roman when available, with a serif fallback.
Captions stay in the article.

To check generation without replacing the supplied figures:

```sh
uv run python -m utils.make_main_figures --output-dir /tmp/async-rl-main-figures
uv run python -m utils.make_section_illustrations --output-dir /tmp/async-rl-section-figures
```

The earlier response-length GIFs and appendix toy PNGs are supplied assets;
their generators are not in this repository.

## Quarto

Install [Quarto](https://quarto.org/docs/get-started/) at the version listed in
[.quarto-version](.quarto-version). From the repository root:

```sh
./scripts/build-website --preview  # local preview with live updates
./scripts/build-website            # build HTML in _site/
```

The build replaces `_site/`. To compare themes, run
`./scripts/build-website --all`; open `_site/index.html` for the theme picker.

The [Pages workflow](.github/workflows/pages.yml) builds the site on pushes to
`main` and supports manual runs. Publishing also requires GitHub Pages to be
enabled in the repository settings, with GitHub Actions as the source.

## Tests, style, linting

```sh
uv run python -m unittest discover -s src/tests -v
uv run black --check src
uv run ruff check src
```

The tests cover response moments, memory, integer and continuous allocation,
and the finite-batch timing simulation. Run `uv run black src` to format code.

## Optional timing experiments

```sh
uv run python -m utils.make_batch_timing_preview
uv run python -m utils.make_batch_timing_sweep
```

These write mean and 99th-percentile timing predictions to `figs/review/`.
The preview varies sequence length; the sweep varies GPU counts, TP and batches.
Both accept `--output-dir`; use `--help` for simulation settings.

## Citation

```bibtex
@misc{guenais2026asyncrl,
  title  = {An analytical model of (async) {RL} systems},
  author = {Guenais, Theo},
  year   = {2026},
  url    = {https://tguens.github.io/scratch-rl-notes/}
}
```

## License

Code: [MIT](LICENSE). Text and figures (`docs/`, `figs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

[Theo Guenais](https://github.com/tguens)
