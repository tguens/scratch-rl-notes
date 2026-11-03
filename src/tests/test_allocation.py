"""Numerical checks of distributions, feasibility and the integer optimum."""

import math
import unittest

from allocation import MODELS, _attention, _statistics, allocate, response_law


class AllocationTests(unittest.TestCase):
    def test_capped_discrete_means(self):
        for arrival in ("geometric", "lognormal"):
            for mean, maximum in ((1, 8), (3.7, 12), (8, 8), (1024, 3584)):
                law = response_law(arrival, mean, maximum)
                self.assertAlmostEqual(sum(p for _, p in law), 1, places=12)
                self.assertAlmostEqual(sum(t * p for t, p in law), mean, places=8)
                self.assertLessEqual(max(t for t, _ in law), maximum)

    def test_context_limit(self):
        with self.assertRaises(ValueError):
            allocate(8, "OLMo-3-7B", mean_response=2048, max_response=65536)

    def test_degenerate_random_lengths_keep_the_random_age_reserve(self):
        from continuous_allocation import continuous

        for arrival in ("geometric", "lognormal"):
            for mode in ("fluid", "cohorts"):
                result = continuous(
                    32,
                    arrival=arrival,
                    mean_response=16384,
                    max_response=16384,
                    memory_fraction=0.9,
                    staggered_memory=mode,
                )
                r = result["integer_solution"]
                m = MODELS[r["model"]]
                DQ, DKV = m["NQ"] * m["d"], m["NKV"] * m["d"]
                P = m["L"] * (3 * m["D"] * m["F"] + 2 * m["D"] * DQ + 2 * m["D"] * DKV)
                st, B = r["statistics"], r["batch_per_replica"]
                self.assertGreater(st["kv_var"], 0)
                reserve = B * st["kv_mean"] + 3 * math.sqrt(B * st["kv_var"])
                self.assertAlmostEqual(
                    r["sampler_memory_bytes"],
                    (2 * P + reserve) / r["sampler_tp"],
                    places=4,
                )
                self.assertGreaterEqual(
                    result["sequences_per_second"] + 1e-12, r["sequences_per_second"]
                )

    def test_olmo3_windowed_cache(self):
        from allocation import _kv

        m = MODELS["OLMo-3-7B"]
        self.assertEqual(m["max_context"], 65536)
        self.assertEqual(_kv(m, 65536), 4 * 32 * 128 * (8 * 65536 + 24 * 4096))
        self.assertEqual(
            _attention(m, 65536),
            8 * 65536 * 65537 / 2 + 24 * (4096 * 4097 / 2 + 61440 * 4096),
        )

    def test_unused_gpus_are_allowed(self):
        r = allocate(
            7, mean_response=512, max_response=1024, sampler_tp=(2,), trainer_tp=(2,)
        )
        self.assertTrue(r["feasible"])
        self.assertEqual(r["unused_gpus"], 1)

    def test_sync_and_epochs_change_the_objective_consistently(self):
        a = allocate(16, mean_response=512, max_response=2048, weight_sync=False)
        b = allocate(16, mean_response=512, max_response=2048, weight_sync=True)
        self.assertGreaterEqual(
            a["sequences_per_second"] + 1e-12, b["sequences_per_second"]
        )
        c = allocate(16, mean_response=512, max_response=2048, epochs=2)
        self.assertGreaterEqual(
            b["sequences_per_second"] + 1e-12, c["sequences_per_second"]
        )

    def test_invalid_inputs(self):
        for kwargs in (
            dict(n_gpus=1),
            dict(n_gpus=8, epochs=0),
            dict(n_gpus=8, sampler_tp=(3,)),
            dict(n_gpus=8, mean_staleness_limit=1),
            dict(n_gpus=8, memory_fraction=1.1),
            dict(n_gpus=8, trainer_efficiency=0),
            dict(n_gpus=8, trainer_efficiency=1.1),
            dict(n_gpus=8, trainer_efficiency=math.nan),
            dict(n_gpus=8, sampler_efficiency=0),
            dict(n_gpus=8, sampler_efficiency=1.1),
        ):
            with self.assertRaises(ValueError):
                allocate(**kwargs)

    def test_trainer_efficiency_applies_only_to_training(self):
        # One feasible split and B=1 remove layout changes from this check.
        m = dict(
            name="tiny",
            L=2,
            D=16,
            F=32,
            NQ=4,
            NKV=2,
            d=4,
            max_context=64,
            window=0,
            global_every=1,
        )
        hw = dict(
            name="tiny",
            C=2000.0,
            W=1000.0,
            HBM=100000.0,
            W_ici=500.0,
            W_link=200.0,
            gpus_per_node=8,
        )
        kwargs = dict(
            mean_response=2.5,
            max_response=8,
            prompt_tokens=2,
            n_sequences=1,
            mean_staleness_limit=2,
            sampler_tp=(1,),
            trainer_tp=(1,),
        )
        base = allocate(2, m, hw, **kwargs)
        slow = allocate(2, m, hw, trainer_efficiency=0.4, **kwargs)
        self.assertAlmostEqual(slow["trainer_seconds"], base["trainer_seconds"] / 0.4)
        self.assertEqual(slow["trainer_model_seconds"], base["trainer_model_seconds"])
        for key in (
            "sampler_seconds",
            "sync_seconds",
            "sampler_memory_bytes",
            "trainer_memory_bytes",
        ):
            self.assertEqual(slow[key], base[key])
        self.assertEqual(
            slow["cycle_seconds"],
            max(
                slow["trainer_seconds"], slow["sampler_seconds"] + slow["sync_seconds"]
            ),
        )
        slow_sampler = allocate(2, m, hw, sampler_efficiency=0.25, **kwargs)
        self.assertEqual(slow_sampler["trainer_seconds"], base["trainer_seconds"])
        self.assertEqual(slow_sampler["sync_seconds"], base["sync_seconds"])
        self.assertAlmostEqual(
            slow_sampler["sampler_seconds"], 4 * base["sampler_seconds"]
        )
        self.assertEqual(
            slow_sampler["sampler_model_seconds"], base["sampler_model_seconds"]
        )

    def test_common_efficiency_preserves_split_without_sync(self):
        kwargs = dict(
            mean_response=512, max_response=512, arrival="staggered", weight_sync=False
        )
        base = allocate(16, **kwargs)
        slow = allocate(16, sampler_efficiency=0.5, trainer_efficiency=0.5, **kwargs)
        for key in (
            "sampler_gpus",
            "trainer_gpus",
            "sampler_tp",
            "trainer_tp",
            "batch_per_replica",
        ):
            self.assertEqual(slow[key], base[key])
        self.assertAlmostEqual(slow["cycle_seconds"], 2 * base["cycle_seconds"])

    def test_cache_and_training_attention_use_local_windows(self):
        local = MODELS["Gemma-3-12B"]
        full = dict(local, window=0, global_every=1)
        a = _statistics(
            tuple(sorted(local.items())), "geometric", 2048.0, 3584, 512, 1.0
        )
        b = _statistics(
            tuple(sorted(full.items())), "geometric", 2048.0, 3584, 512, 1.0
        )
        self.assertLess(a["kv_mean"], b["kv_mean"])
        self.assertLess(a["attention_pairs"], b["attention_pairs"])
        toy = dict(L=2, global_every=2, window=2)
        self.assertEqual(_attention(toy, 3), 6 + 5)

    def test_renewal_cache_matches_analytical_moments(self):
        m = MODELS["Llama-3.1-8B"]
        kappa = 4 * m["L"] * m["NKV"] * m["d"]
        for arrival in ("staggered", "geometric", "lognormal"):
            law = response_law(arrival, 5.0, 20)
            et = math.fsum(t * p for t, p in law)
            age_mean = math.fsum(t * (t + 1) * p for t, p in law) / (2 * et)
            age_second = math.fsum(t * (t + 1) * (2 * t + 1) * p for t, p in law) / (
                6 * et
            )
            stats = _statistics(tuple(sorted(m.items())), arrival, 5.0, 20, 3, 1.0)
            self.assertAlmostEqual(stats["kv_mean"] / kappa, 3 + age_mean, places=11)
            self.assertAlmostEqual(
                stats["kv_var"] / kappa**2, age_second - age_mean**2, places=10
            )

    def test_feasibility_and_node_placement(self):
        for hardware in ("H100", "H200", "B200"):
            for model in MODELS:
                r = allocate(
                    32,
                    model,
                    hardware,
                    mean_response=2048,
                    max_response=3584,
                    memory_fraction=0.9,
                    mean_staleness_limit=2.5,
                )
                self.assertTrue(r["feasible"])
                self.assertLessEqual(
                    r["estimated_mean_first_token_staleness"], 2.5 + 1e-12
                )
                self.assertLessEqual(
                    r["sampler_memory_bytes"], r["usable_memory_bytes"] * (1 + 1e-12)
                )
                self.assertLessEqual(
                    r["trainer_memory_bytes"], r["usable_memory_bytes"] * (1 + 1e-12)
                )
                self.assertEqual(r["sampler_gpus"], r["replicas"] * r["sampler_tp"])
                self.assertEqual(r["trainer_gpus"], r["fsdp_ranks"] * r["trainer_tp"])
                seen = []
                for node in r["placement"]:
                    for group in node["groups"]:
                        ids = group["gpu_ids"]
                        self.assertEqual(ids[0] // 8, ids[-1] // 8)
                        seen.extend(ids)
                self.assertEqual(len(seen), len(set(seen)))
                self.assertEqual(len(seen) + r["unused_gpus"], 32)

    def test_custom_node_size_tries_smaller_trainer_pools(self):
        # 8 trainer GPUs do not fit beside a TP4 sampler in two 6-GPU nodes; 4 do.
        hw = dict(
            C=989.5e12, W=3.35e12, HBM=80e9, W_ici=450e9, W_link=50e9, gpus_per_node=6
        )
        r = allocate(12, hardware=hw, sampler_tp=(4,), trainer_tp=(4,))
        self.assertTrue(r["feasible"])
        self.assertEqual((r["sampler_gpus"], r["trainer_gpus"]), (4, 4))

    def test_lognormal_fit_with_large_sigma(self):
        for mean in (2, 7.5):
            law = response_law("lognormal", mean, 8, 50.0)
            self.assertAlmostEqual(sum(t * p for t, p in law), mean, places=8)

    def test_integer_minimum_matches_independent_exhaustive_search(self):
        m = dict(
            name="tiny",
            L=2,
            D=16,
            F=32,
            NQ=4,
            NKV=2,
            d=4,
            max_context=64,
            window=0,
            global_every=1,
        )
        hw = dict(
            name="tiny",
            C=2000.0,
            W=1000.0,
            HBM=100000.0,
            W_ici=500.0,
            W_link=200.0,
            gpus_per_node=8,
        )
        st = _statistics(tuple(sorted(m.items())), "geometric", 2.5, 8, 2, 1.0)
        P = 2 * (3 * 16 * 32 + 2 * 16 * 16 + 2 * 16 * 8)
        act = (2 * 2 * 16 + 8 * 16 + 4 * 16 + 4 * 8 + 6 * 32) * st["max_length"]
        for sync in (False, True):
            for epochs, efficiency, sampler_eff in (
                (1, 1.0, 1.0),
                (3, 1.0, 1.0),
                (1, 0.4, 1.0),
                (3, 0.4, 1.0),
                (1, 1.0, 0.4),
                (3, 0.4, 0.2),
            ):
                r = allocate(
                    7,
                    m,
                    hw,
                    mean_response=2.5,
                    max_response=8,
                    prompt_tokens=2,
                    n_sequences=4,
                    mean_staleness_limit=2.3,
                    epochs=epochs,
                    trainer_efficiency=efficiency,
                    sampler_efficiency=sampler_eff,
                    weight_sync=sync,
                    sampler_tp=(1, 2),
                    trainer_tp=(1, 2),
                )
                brute = math.inf
                for s in (1, 2):
                    for replicas in range(1, 8):
                        ns = s * replicas
                        for t in (1, 2):
                            for ranks in range(1, 8):
                                nt = t * ranks
                                if ns + nt > 7 or 16 * P / nt + act / t > hw["HBM"]:
                                    continue
                                train = (
                                    epochs
                                    * 4
                                    / nt
                                    * (
                                        (
                                            8 * P * st["mean_length"]
                                            + 18 * 16 * st["attention_pairs"]
                                        )
                                        / hw["C"]
                                        + 24
                                        * 2
                                        * 16
                                        * st["mean_length"]
                                        * (t - 1)
                                        / hw["W_ici"]
                                    )
                                    / efficiency
                                )
                                for B in range(1, 30):
                                    if 1 + replicas * B / 4 > 2.3 + 1e-12:
                                        continue
                                    memory = (
                                        2 * P
                                        + B * st["kv_mean"]
                                        + 3 * math.sqrt(B * st["kv_var"])
                                    ) / s
                                    if memory > hw["HBM"]:
                                        continue
                                    step = (
                                        max(
                                            2 * B * P / (s * hw["C"]),
                                            2 * P / (s * hw["W"]),
                                        )
                                        + B * st["kv_mean"] / (s * hw["W"])
                                        + 8 * 2 * B * 16 * (s - 1) / (s * hw["W_ici"])
                                    )
                                    gen = (
                                        4
                                        * st["mean_response"]
                                        / (replicas * B)
                                        * step
                                        / sampler_eff
                                    )
                                    pause = 2 * P / (s * hw["W_link"]) if sync else 0
                                    brute = min(brute, max(gen + pause, train))
                self.assertAlmostEqual(r["cycle_seconds"], brute, places=10)


if __name__ == "__main__":
    unittest.main()
