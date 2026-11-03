"""Continuous allocation under the article's timing and memory assumptions.

``continuous`` returns a plain dictionary, including the matched integer result
under ``integer_solution``. Both solvers use the same costs. Counts of replicas
and sequences per replica can be fractional in the continuous result.

Staggered lengths require ``staggered_memory='fluid'``: exact cohort divisibility
has no continuous analogue. This setting budgets mean cache occupancy.
"""

import math

from allocation import (
    HARDWARE,
    MODELS,
    _activation_bytes,
    _kv,
    _model_parameters,
    _statistics,
    _training_work,
    allocate,
)


def positive_root(a, b, c):
    """Solve a*x**2 + b*x + c = 0 for its positive root (a >= 0, c < 0).

    For a == 0, b must be positive. Choose the quadratic expression that avoids
    subtracting nearly equal values.
    """
    if a == 0:
        return -c / b
    discriminant = math.sqrt(b * b - 4 * a * c)
    if b >= 0:
        return -2 * c / (b + discriminant)
    return (-b + discriminant) / (2 * a)


def _batch_candidates(
    n_gpus,
    sampler_tp,
    trainer_floor,
    slot_limit,
    p,
    q,
    trainer_work,
    sync_per_sequence,
    batch_max,
):
    """Endpoints and capacity-bound intersections from Appendix 6."""
    candidates = [1.0, batch_max]
    if math.isfinite(slot_limit):
        slots = sampler_tp * slot_limit
        # This substitution applies to intersections, not to sampler work p/B+q.
        adjusted_p = p + sync_per_sequence * slots
        candidates.append(
            positive_root(
                n_gpus * q,
                n_gpus * adjusted_p - slots * (q + trainer_work),
                -slots * adjusted_p,
            )
        )
        candidates.append(slots / (n_gpus - trainer_floor))
        if q > 0:
            candidates.append(
                (slots * trainer_work / (n_gpus - sampler_tp) - adjusted_p) / q
            )
    return candidates


def continuous(n_gpus, model="Llama-3.1-8B", hardware="H100", **kwargs):
    """Return the optimal continuous allocation and the matched integer result.

    Model names, hardware names and keyword options match ``allocation.allocate``.
    The integer solver validates inputs and supplies the normalized options.
    Check ``result['feasible']`` before reading allocation fields.
    """
    integer = allocate(n_gpus, model, hardware, **kwargs)
    inputs = integer["inputs"]
    if inputs["arrival"] == "staggered" and inputs["staggered_memory"] != "fluid":
        raise ValueError(
            "Continuous staggered comparison requires staggered_memory='fluid'"
        )

    m = MODELS[model] if isinstance(model, str) else model
    hw = HARDWARE[hardware] if isinstance(hardware, str) else hardware
    statistics = _statistics(
        tuple(sorted(m.items())),
        inputs["arrival"],
        float(inputs["mean_response"]),
        inputs["max_response"],
        inputs["prompt_tokens"],
        inputs["lognormal_sigma"],
    )
    compute = hw["C"] * inputs["compute_efficiency"]
    bandwidth = hw["W"] * inputs["bandwidth_efficiency"]
    memory = hw["HBM"] * inputs["memory_fraction"]
    parameters = _model_parameters(m)
    activations = _activation_bytes(m, statistics["max_length"])
    n_sequences = inputs["n_sequences"]
    slot_limit = (
        math.inf
        if inputs["mean_staleness_limit"] is None
        else (inputs["mean_staleness_limit"] - 1) * n_sequences
    )
    sampler_efficiency = inputs["sampler_efficiency"]
    best = None

    for sampler_tp in inputs["sampler_tp_choices"]:
        cache_bytes = sampler_tp * memory - 2 * parameters
        if cache_bytes <= 0:
            continue
        mean_cache = statistics["kv_mean"]
        cache_spread = inputs["cache_z"] * math.sqrt(statistics["kv_var"])
        if inputs["arrival"] in ("synchronized", "staggered"):
            if inputs["arrival"] == "synchronized":
                mean_cache = _kv(m, statistics["max_length"])
            cache_spread = 0.0
        # Solve mean_cache*B + cache_spread*sqrt(B) <= cache_bytes.
        sqrt_batch = (
            2
            * cache_bytes
            / (cache_spread + math.sqrt(cache_spread**2 + 4 * mean_cache * cache_bytes))
        )
        batch_max = min(sqrt_batch**2, slot_limit, max(1.0, compute / bandwidth))
        if batch_max < 1:
            continue

        sync_per_sequence = (
            2 * parameters / (sampler_tp * hw["W_link"] * n_sequences)
            if inputs["weight_sync"]
            else 0.0
        )
        # On the useful batch range, sampler GPU-seconds per sequence = p/B + q.
        p = (
            2
            * parameters
            * statistics["mean_response"]
            / (bandwidth * sampler_efficiency)
        )
        q = (
            statistics["mean_response"]
            * (
                statistics["kv_mean"] / bandwidth
                + 8 * m["L"] * m["D"] * (sampler_tp - 1) / hw["W_ici"]
            )
            / sampler_efficiency
        )
        if compute / bandwidth < 1:
            # Even B=1 is compute bound; no weight-sharing gain remains.
            p = 0.0
            q += (
                2
                * parameters
                * statistics["mean_response"]
                / (compute * sampler_efficiency)
            )

        for trainer_tp in inputs["trainer_tp_choices"]:
            state_memory = memory - activations / trainer_tp
            if state_memory <= 0:
                continue
            trainer_floor = max(trainer_tp, 16 * parameters / state_memory)
            if n_gpus < sampler_tp + trainer_floor:
                continue
            trainer_work = (
                _training_work(
                    m, statistics, compute, hw["W_ici"], trainer_tp, inputs["epochs"]
                )
                / inputs["trainer_efficiency"]
            )

            candidates = _batch_candidates(
                n_gpus,
                sampler_tp,
                trainer_floor,
                slot_limit,
                p,
                q,
                trainer_work,
                sync_per_sequence,
                batch_max,
            )
            for batch in candidates:
                if not 1 - 1e-10 <= batch <= batch_max + 1e-10:
                    continue
                batch = min(batch_max, max(1.0, batch))
                sampler_work = p / batch + q
                sampler_max = min(
                    n_gpus - trainer_floor, sampler_tp * slot_limit / batch
                )
                if sampler_max < sampler_tp - 1e-10:
                    continue
                balanced_sampler = positive_root(
                    sync_per_sequence,
                    sampler_work + trainer_work - sync_per_sequence * n_gpus,
                    -n_gpus * sampler_work,
                )
                sampler_gpus = min(sampler_max, max(sampler_tp, balanced_sampler))
                trainer_gpus = n_gpus - sampler_gpus
                rate = min(
                    sampler_gpus / (sampler_work + sync_per_sequence * sampler_gpus),
                    trainer_gpus / trainer_work,
                )
                if best is not None and rate <= best["sequences_per_second"]:
                    continue
                best = dict(
                    feasible=True,
                    n_gpus=n_gpus,
                    sampler_gpus=sampler_gpus,
                    trainer_gpus=trainer_gpus,
                    sampler_tp=sampler_tp,
                    trainer_tp=trainer_tp,
                    batch_per_replica=batch,
                    sequences_per_second=rate,
                    trainer_floor_gpus=trainer_floor,
                    trainer_at_memory_floor=math.isclose(
                        trainer_gpus, trainer_floor, rel_tol=1e-8
                    ),
                    staleness_binding=math.isfinite(slot_limit)
                    and math.isclose(
                        sampler_gpus * batch / sampler_tp, slot_limit, rel_tol=1e-8
                    ),
                    sampler_seconds=n_sequences * sampler_work / sampler_gpus,
                    trainer_seconds=n_sequences * trainer_work / trainer_gpus,
                    sync_seconds=sync_per_sequence * n_sequences,
                )

    if best is None:
        best = dict(feasible=False)
    best["integer_solution"] = integer
    best["inputs"] = inputs
    return best
