"""Integer allocation for the article's mean-cost model. Standard library only.

    from allocation import allocate
    result = allocate(32, model="Llama-3.1-8B", hardware="H200")

Times are roofline predictions, not GPU measurements. Staleness is an estimated
mean over sequences, not a per-token guarantee. See _appendix.md for assumptions.
"""

import math
from bisect import bisect_right
from functools import lru_cache

# Dense BF16, decimal bytes, one-way NVLink. W_link is an assumed effective
# receiving rate, not the aggregate node network bandwidth.
HARDWARE = {
    "H100": dict(
        C=989.5e12, W=3.35e12, HBM=80e9, W_ici=450e9, W_link=50e9, gpus_per_node=8
    ),
    "H200": dict(
        C=989.5e12, W=4.8e12, HBM=141e9, W_ici=450e9, W_link=50e9, gpus_per_node=8
    ),
    "B200": dict(
        C=2250e12, W=8e12, HBM=180e9, W_ici=900e9, W_link=50e9, gpus_per_node=8
    ),
}

# Text transformer blocks only: no embeddings, LM head, or vision encoder.
MODELS = {
    "Llama-3.1-8B": dict(
        L=32,
        D=4096,
        F=14336,
        NQ=32,
        NKV=8,
        d=128,
        max_context=131072,
        window=0,
        global_every=1,
    ),
    "Qwen3-8B": dict(
        L=36,
        D=4096,
        F=12288,
        NQ=32,
        NKV=8,
        d=128,
        max_context=40960,
        window=0,
        global_every=1,
    ),
    "OLMo-3-7B": dict(
        L=32,
        D=4096,
        F=11008,
        NQ=32,
        NKV=32,
        d=128,
        max_context=65536,
        window=4096,
        global_every=4,
    ),
    "Gemma-3-12B": dict(
        L=48,
        D=3840,
        F=15360,
        NQ=16,
        NKV=8,
        d=256,
        max_context=131072,
        window=1024,
        global_every=6,
    ),
}


@lru_cache(maxsize=128)
def response_law(arrival, mean, maximum, sigma=1.0):
    """Return (length, probability) pairs; the *capped discrete* mean is `mean`.

    Log-normal lengths are rounded to the nearest positive integer, then capped.
    Geometric and log-normal parameters are tuned after applying the cap.
    """
    if not (math.isfinite(mean) and 1 <= mean <= maximum):
        raise ValueError("mean_response must be between 1 and max_response")
    if arrival not in ("synchronized", "staggered", "geometric", "lognormal"):
        raise ValueError(
            "arrival must be synchronized, staggered, geometric, or lognormal"
        )
    if arrival in ("synchronized", "staggered"):
        if mean != int(mean):
            raise ValueError("Equal-length responses require an integer mean")
        return ((int(mean), 1.0),)
    if mean == 1 or mean == maximum:
        return ((int(mean), 1.0),)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("lognormal_sigma must be positive")

    def survival(parameter):
        if arrival == "geometric":
            # parameter = log of the uncut mean. P(T >= j) = (1-p)^(j-1).
            p = math.exp(-parameter)
            log_q = math.log1p(-p) if p < 1 else -math.inf
            return [1.0] + [math.exp((j - 1) * log_q) for j in range(2, maximum + 1)]
        # parameter = mean of log(uncut continuous length).
        denom = sigma * math.sqrt(2)
        return [1.0] + [
            0.5 * math.erfc((math.log(j - 0.5) - parameter) / denom)
            for j in range(2, maximum + 1)
        ]

    lo, hi = (-40.0, math.log(maximum) + 40.0)
    if arrival == "geometric":
        lo = 0.0
    # A large sigma needs a wider bracket. Widen it until it contains the mean.
    steps = 55
    while lo < 0 and math.fsum(survival(lo)) > mean and steps < 2000:
        lo, steps = 2 * lo, steps + 1
    while math.fsum(survival(hi)) < mean and steps < 2000:
        hi, steps = 2 * hi, steps + 1
    for _ in range(steps):
        mid = (lo + hi) / 2
        if math.fsum(survival(mid)) < mean:
            lo = mid
        else:
            hi = mid
    sf = survival((lo + hi) / 2)
    if abs(math.fsum(sf) - mean) > 1e-6 * mean:
        raise ValueError("could not fit the capped mean; reduce lognormal_sigma")
    probabilities = [max(0.0, a - b) for a, b in zip(sf, sf[1:] + [0.0])]
    return tuple((j, p) for j, p in enumerate(probabilities, 1) if p > 0)


def _kv(model, context):
    """Used KV bytes per sequence; local-attention layers evict old entries."""
    global_layers = model["L"] // model["global_every"]
    local_layers = model["L"] - global_layers
    attended = global_layers * context + local_layers * min(context, model["window"])
    return 4 * model["NKV"] * model["d"] * attended


def _attention(model, length):
    """Causal attention-pair count, summed over layers, including the diagonal."""
    global_layers = model["L"] // model["global_every"]
    local_layers = model["L"] - global_layers
    full = length * (length + 1) / 2
    window = model["window"]
    local = (
        full
        if length <= window
        else window * (window + 1) / 2 + (length - window) * window
    )
    return global_layers * full + local_layers * local


@lru_cache(maxsize=256)
def _statistics(model_items, arrival, mean, maximum, prompt, sigma):
    model = dict(model_items)
    law = response_law(arrival, mean, maximum, sigma)
    et = math.fsum(t * p for t, p in law)
    et2 = math.fsum(t * t * p for t, p in law)
    largest = max(t for t, _ in law)
    probabilities = dict(law)
    sf, survival = [0.0] * (largest + 1), 0.0
    for j in range(largest, 0, -1):
        survival += probabilities.get(j, 0.0)
        sf[j] = survival
    # Sampling a slot at a decode step weights age j by P(T >= j) / E[T].
    kv_mean = (
        math.fsum(sf[j] * _kv(model, prompt + j) for j in range(1, largest + 1)) / et
    )
    kv_var = (
        math.fsum(
            sf[j] * (_kv(model, prompt + j) - kv_mean) ** 2
            for j in range(1, largest + 1)
        )
        / et
    )
    return dict(
        mean_response=et,
        second_response_moment=et2,
        mean_length=prompt + et,
        second_length_moment=prompt**2 + 2 * prompt * et + et2,
        max_length=prompt + largest,
        kv_mean=kv_mean,
        kv_var=kv_var,
        attention_pairs=math.fsum(p * _attention(model, prompt + t) for t, p in law),
        deterministic=len(law) == 1,
        response_length=largest,
    )


def _model_parameters(model):
    """Transformer-block parameters: gated MLP and Q/K/V/output projections."""
    D, F = model["D"], model["F"]
    DQ, DKV = model["NQ"] * model["d"], model["NKV"] * model["d"]
    return model["L"] * (3 * D * F + 2 * D * DQ + 2 * D * DKV)


def _activation_bytes(model, max_length):
    """Full-recompute activation budget for one sequence before TP/SP sharding."""
    D, F = model["D"], model["F"]
    DQ, DKV = model["NQ"] * model["d"], model["NKV"] * model["d"]
    return (2 * model["L"] * D + 8 * D + 4 * DQ + 4 * DKV + 6 * F) * max_length


def _training_work(model, statistics, compute, interconnect, tp, epochs):
    """Modeled trainer GPU-seconds per fresh sequence, before stage efficiency.

    c=4: forward, backward and full recompute. Attention includes causal diagonal
    entries and each layer's local window. Once-per-batch reference passes are
    outside this cost.
    """
    parameters = _model_parameters(model)
    query_width = model["NQ"] * model["d"]
    return epochs * (
        (
            8 * parameters * statistics["mean_length"]
            + 18 * query_width * statistics["attention_pairs"]
        )
        / compute
        + 24
        * model["L"]
        * model["D"]
        * statistics["mean_length"]
        * (tp - 1)
        / interconnect
    )


def _placement(n_gpus, gpus_per_node, sampler_tp, replicas, trainer_tp, fsdp_ranks):
    """Pack power-of-two TP groups wholly inside nodes; pools may share a node."""
    nodes = [
        dict(node=i, capacity=min(gpus_per_node, n_gpus - i * gpus_per_node), groups=[])
        for i in range(math.ceil(n_gpus / gpus_per_node))
    ]
    used = [0] * len(nodes)
    first_fit = {}
    groups = [(sampler_tp, "sampler")] * replicas + [
        (trainer_tp, "trainer")
    ] * fsdp_ranks
    for size, role in sorted(groups, reverse=True):
        # A node skipped for this size cannot acquire more free space later.
        for index in range(first_fit.get(size, 0), len(nodes)):
            node = nodes[index]
            if used[index] + size <= node["capacity"]:
                node["groups"].append(
                    dict(
                        role=role,
                        size=size,
                        gpu_ids=list(
                            range(
                                node["node"] * gpus_per_node + used[index],
                                node["node"] * gpus_per_node + used[index] + size,
                            )
                        ),
                    )
                )
                used[index] += size
                first_fit[size] = index
                break
        else:
            return None
    return nodes


def allocate(
    n_gpus,
    model="Llama-3.1-8B",
    hardware="H100",
    *,
    mean_response=2048,
    arrival="geometric",
    prompt_tokens=512,
    max_response=None,
    epochs=1,
    n_sequences=1024,
    mean_staleness_limit=8.0,
    weight_sync=True,
    sampler_tp=(1, 2, 4, 8),
    trainer_tp=(1, 2, 4, 8),
    memory_fraction=1.0,
    compute_efficiency=1.0,
    bandwidth_efficiency=1.0,
    sampler_efficiency=1.0,
    trainer_efficiency=1.0,
    lognormal_sigma=1.0,
    cache_z=3.0,
    stagger_groups=64,
    staggered_memory="fluid",
):
    """Return the fastest feasible integer allocation as a plain dict.

    `model` / `hardware` accept a built-in name or a dictionary with the same keys.
    Staleness limit means 1 + replicas*B/n_sequences <= limit; it is not floored.
    Full checkpoints, BF16, one sequence per micro-batch. FSDP communication is
    assumed hidden. Node-wide TP is not forced: TP is searched locally.
    Each stage efficiency = modeled / actual time at the same layout;
    its time is divided by that ratio. These are not compute MFU. Sync is separate.
    staggered_memory='fluid' budgets mean KV; 'cohorts' reserves the exact peak.
    """
    if not isinstance(n_gpus, int) or isinstance(n_gpus, bool) or n_gpus < 2:
        raise ValueError("n_gpus must be an integer >= 2")
    for name, value in (
        ("epochs", epochs),
        ("n_sequences", n_sequences),
        ("max_response", max_response),
        ("stagger_groups", stagger_groups),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(name + " must be a positive integer")
    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        raise ValueError("prompt_tokens must be a nonnegative integer")
    if staggered_memory not in ("cohorts", "fluid"):
        raise ValueError("staggered_memory must be cohorts or fluid")
    for value in (
        memory_fraction,
        compute_efficiency,
        bandwidth_efficiency,
        sampler_efficiency,
        trainer_efficiency,
    ):
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Memory fraction and efficiencies must be in (0, 1]")
    if not math.isfinite(cache_z) or cache_z < 0:
        raise ValueError("cache_z must be nonnegative")
    if mean_staleness_limit is not None and (
        not math.isfinite(mean_staleness_limit) or mean_staleness_limit <= 1
    ):
        raise ValueError("mean_staleness_limit must exceed 1, or be None")
    model_name = model if isinstance(model, str) else model.get("name", "custom")
    hardware_name = (
        hardware if isinstance(hardware, str) else hardware.get("name", "custom")
    )
    m = MODELS[model] if isinstance(model, str) else model
    hw = HARDWARE[hardware] if isinstance(hardware, str) else hardware
    available_context = m["max_context"] - prompt_tokens
    maximum = (
        max_response
        if max_response is not None
        else min(math.ceil(4 * mean_response), available_context)
    )
    if maximum < 1 or maximum > available_context:
        raise ValueError("prompt_tokens + max_response exceeds the model context limit")
    st = _statistics(
        tuple(sorted(m.items())),
        arrival,
        float(mean_response),
        maximum,
        prompt_tokens,
        float(lognormal_sigma),
    )
    C, W = hw["C"] * compute_efficiency, hw["W"] * bandwidth_efficiency
    hbm = hw["HBM"] * memory_fraction
    L, D = m["L"], m["D"]
    P = _model_parameters(m)
    act = _activation_bytes(m, st["max_length"])
    node_size = hw["gpus_per_node"]

    def valid_tps(choices):
        result = []
        for tp in choices:
            if not isinstance(tp, int) or tp < 1 or tp & (tp - 1):
                raise ValueError("TP sizes must be positive powers of two")
            if tp <= node_size and m["NQ"] % tp == 0 and m["NKV"] % tp == 0:
                result.append(tp)
        return sorted(set(result))

    def generation_model(B, tp, nrep):
        step = (
            max(2 * B * P / C, 2 * P / W)
            + B * st["kv_mean"] / W
            + 8 * L * B * D * (tp - 1) / hw["W_ici"]
        ) / tp
        return n_sequences * st["mean_response"] * step / (nrep * B)

    def generation(B, tp, nrep):
        return generation_model(B, tp, nrep) / sampler_efficiency

    def sampler_memory(B, tp):
        if arrival == "staggered" and staggered_memory == "fluid":
            used_kv = B * st["kv_mean"]
        elif arrival in ("synchronized", "staggered"):
            J = st["response_length"]
            groups = (
                1
                if arrival == "synchronized"
                else max(
                    g
                    for g in range(1, min(stagger_groups, B, J) + 1)
                    if B % g == 0 and J % g == 0
                )
            )
            used_kv = (
                B
                * math.fsum(
                    _kv(m, prompt_tokens + k * J // groups)
                    for k in range(1, groups + 1)
                )
                / groups
            )
        else:
            # A point-mass response law still has spread in stationary slot ages.
            used_kv = B * st["kv_mean"] + cache_z * math.sqrt(B * st["kv_var"])
        return (2 * P + used_kv) / tp

    sampler_sizes = valid_tps(sampler_tp)
    trainer_sizes = valid_tps(trainer_tp)
    # Above this batch size linear layers are compute-bound; larger B gives no
    # extra throughput in this model. A prefix of feasible batches also handles
    # exact deterministic cohort divisibility without a non-monotone binary search.
    # Exact cohort divisibility can make a slightly larger compute-bound batch
    # fit when a nearby smaller one does not. Check the next group multiple too.
    search_limit = math.ceil(C / W)
    if arrival == "staggered" and st["deterministic"] and staggered_memory == "cohorts":
        search_limit += stagger_groups - 1
    batches = {
        tp: [
            B
            for B in range(1, search_limit + 1)
            if sampler_memory(B, tp) <= hbm * (1 + 1e-12)
        ]
        for tp in sampler_sizes
    }
    best, best_key = None, None
    for s, feasible_batches in batches.items():
        if not feasible_batches:
            continue
        sync = 2 * P / (s * hw["W_link"]) if weight_sync else 0.0
        for nsamp in range(s, n_gpus, s):
            nrep = nsamp // s
            stale_cap = (
                math.inf
                if mean_staleness_limit is None
                else ((mean_staleness_limit - 1) * n_sequences / nrep)
            )
            index = bisect_right(feasible_batches, stale_cap + 1e-10) - 1
            if index < 0:
                continue
            B = feasible_batches[index]
            gen = generation(B, s, nrep)
            for t in trainer_sizes:
                # The largest trainer pool is the fastest. With custom node sizes
                # it may not fit into whole nodes, so try smaller pools too.
                placement = None
                for ntrain in range((n_gpus - nsamp) // t * t, t - 1, -t):
                    if 16 * P / ntrain + act / t > hbm * (1 + 1e-12):
                        break  # smaller pools need more memory per GPU
                    train_model = (
                        n_sequences
                        / ntrain
                        * _training_work(m, st, C, hw["W_ici"], t, epochs)
                    )
                    train = train_model / trainer_efficiency
                    cycle = max(gen + sync, train)
                    # If training limits the rate, reduce B while preserving the
                    # same cycle time. This reduces memory and estimated staleness.
                    chosen = next(
                        b
                        for b in feasible_batches[: index + 1]
                        if generation(b, s, nrep) + sync <= cycle * (1 + 1e-12)
                    )
                    staleness = 1 + nrep * chosen / n_sequences
                    key = (cycle, staleness, nsamp, s, t)
                    if best_key is not None and key >= best_key:
                        break  # smaller pools are not faster
                    placement = _placement(n_gpus, node_size, s, nrep, t, ntrain // t)
                    if placement is not None:
                        break
                if placement is None:
                    continue
                best_key = key
                best = dict(
                    feasible=True,
                    model=model_name,
                    hardware=hardware_name,
                    n_gpus=n_gpus,
                    sampler_gpus=nsamp,
                    trainer_gpus=ntrain,
                    unused_gpus=n_gpus - nsamp - ntrain,
                    sampler_tp=s,
                    trainer_tp=t,
                    replicas=nrep,
                    fsdp_ranks=ntrain // t,
                    batch_per_replica=chosen,
                    sequences_per_second=n_sequences / cycle,
                    cycle_seconds=cycle,
                    sampler_seconds=generation(chosen, s, nrep),
                    sync_seconds=sync,
                    sampler_model_seconds=generation_model(chosen, s, nrep),
                    trainer_seconds=train,
                    trainer_model_seconds=train_model,
                    trainer_wait_fraction=1 - train / cycle,
                    trainer_floor_gpus=t
                    * max(1, math.ceil(16 * P / (hbm - act / t) / t - 1e-12)),
                    trainer_floor_reason=(
                        "memory" if 16 * P / (hbm - act / t) > t else "one TP group"
                    ),
                    trainer_at_memory_floor=ntrain
                    == t * max(1, math.ceil(16 * P / (hbm - act / t) / t - 1e-12)),
                    batch_reduced_for_same_rate=chosen < B,
                    estimated_mean_first_token_staleness=staleness,
                    sampler_memory_bytes=sampler_memory(chosen, s),
                    trainer_memory_bytes=16 * P / ntrain + act / t,
                    usable_memory_bytes=hbm,
                    transformer_parameters=P,
                    statistics=dict(st),
                    placement=placement,
                )
    inputs = dict(
        mean_response=mean_response,
        arrival=arrival,
        max_response=maximum,
        prompt_tokens=prompt_tokens,
        epochs=epochs,
        n_sequences=n_sequences,
        mean_staleness_limit=mean_staleness_limit,
        weight_sync=weight_sync,
        memory_fraction=memory_fraction,
        compute_efficiency=compute_efficiency,
        sampler_efficiency=sampler_efficiency,
        trainer_efficiency=trainer_efficiency,
        bandwidth_efficiency=bandwidth_efficiency,
        cache_z=cache_z,
        lognormal_sigma=lognormal_sigma,
        stagger_groups=stagger_groups,
        staggered_memory=staggered_memory,
        sampler_tp_choices=sampler_sizes,
        trainer_tp_choices=trainer_sizes,
    )
    if best is None:
        return dict(
            feasible=False,
            model=model_name,
            hardware=hardware_name,
            n_gpus=n_gpus,
            inputs=inputs,
            reason="No layout fits the memory, TP, GPU-budget, and estimated-mean-staleness constraints",
        )
    best["inputs"] = inputs
    best["assumptions"] = [
        "Mean roofline capacity; not a measured throughput or queueing guarantee",
        "BF16 transformer blocks only; embeddings, LM head, prefill and vision are omitted",
        "FSDP communication hidden; TP groups fit within a node",
        "Each stage time optionally inflated by its modeled/actual ratio; sync is separate; not measured MFU",
        "Used local-attention KV is evicted outside the configured window",
        "Weight reception parallel; trainer export/send and sender fan-out are omitted",
        "Staleness is the Section 3 overlap estimate, not a hard per-token limit",
        "Random cache budget is mean + z*std, not an exact or hard memory quantile",
        "Staggered fluid memory uses mean KV; cohorts reserve the exact configured peak",
    ]
    return best


if __name__ == "__main__":
    import json

    print(json.dumps(allocate(32, hardware="H200"), indent=2))


def continuous(n_gpus, model="Llama-3.1-8B", hardware="H100", **kwargs):
    """Continuous Section 4 relaxation plus the matched integer solution.

    Use staggered_memory='fluid' when arrival='staggered'.
    """
    from continuous_allocation import continuous as solve

    return solve(n_gpus, model, hardware, **kwargs)
