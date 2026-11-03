"""Check the analytical relaxation against integer results and an independent grid."""

import math
import unittest

from allocation import _statistics
from continuous_allocation import continuous, positive_root


class ContinuousTests(unittest.TestCase):
    def test_stable_root(self):
        for a in (0.0, 1e-15, 1.0, 1e15):
            for b in (-1e9, 1.0, 1e9):
                if a == 0 and b < 0:
                    continue
                x = positive_root(a, b, -3.0)
                self.assertGreater(x, 0)
                self.assertLess(
                    abs(a * x * x + b * x - 3.0) / (abs(a * x * x) + abs(b * x) + 3.0),
                    1e-14,
                )

    def test_relaxation_bounds_integer(self):
        for arrival in ("staggered", "geometric", "synchronized"):
            for sync in (True, False):
                for N in (8, 32):
                    r = continuous(
                        N,
                        mean_response=2048,
                        max_response=3584,
                        arrival=arrival,
                        staggered_memory="fluid",
                        weight_sync=sync,
                        memory_fraction=0.9,
                        sampler_efficiency=0.6,
                        trainer_efficiency=0.5,
                    )
                    self.assertTrue(r["feasible"])
                    self.assertGreaterEqual(
                        r["sequences_per_second"] + 1e-10,
                        r["integer_solution"]["sequences_per_second"],
                    )

    def test_continuous_optimum_matches_grid(self):
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
            C=10000.0,
            W=1000.0,
            HBM=100000.0,
            W_ici=500.0,
            W_link=200.0,
            gpus_per_node=8,
        )
        st = _statistics(tuple(sorted(m.items())), "geometric", 2.5, 8, 2, 1.0)
        P = 2 * (3 * 16 * 32 + 2 * 16 * 16 + 2 * 16 * 8)
        act = (2 * 2 * 16 + 8 * 16 + 4 * 16 + 4 * 8 + 6 * 32) * 10
        for sync in (False, True):
            for eta in (1.25, 2.3, 20.0):
                best = 0.0
                r = continuous(
                    7,
                    m,
                    hw,
                    mean_response=2.5,
                    max_response=8,
                    prompt_tokens=2,
                    n_sequences=4,
                    epochs=2,
                    mean_staleness_limit=eta,
                    sampler_efficiency=0.6,
                    trainer_efficiency=0.5,
                    weight_sync=sync,
                    sampler_tp=(1, 2),
                    trainer_tp=(1, 2),
                )
                for s in (1, 2):
                    pause = 2 * P / (s * hw["W_link"]) if sync else 0.0
                    for t in (1, 2):
                        floor = max(t, 16 * P / (hw["HBM"] - act / t))
                        if 7 - floor < s:
                            continue
                        for j in range(201):
                            B = 1 + 9 * j / 200
                            mem = (
                                2 * P
                                + B * st["kv_mean"]
                                + 3 * math.sqrt(B * st["kv_var"])
                            ) / s
                            if mem > hw["HBM"]:
                                continue
                            for k in range(201):
                                ns = s + (7 - floor - s) * k / 200
                                if ns * B / s > (eta - 1) * 4:
                                    continue
                                step = (
                                    max(2 * B * P / hw["C"], 2 * P / hw["W"])
                                    + B * st["kv_mean"] / hw["W"]
                                    + 8 * 2 * B * 16 * (s - 1) / hw["W_ici"]
                                ) / s
                                gen = 4 * 2.5 / (ns / s * B) * step / 0.6
                                train = (
                                    2
                                    * 4
                                    / (7 - ns)
                                    * (
                                        (8 * P * 4.5 + 18 * 16 * st["attention_pairs"])
                                        / hw["C"]
                                        + 24 * 2 * 16 * 4.5 * (t - 1) / hw["W_ici"]
                                    )
                                    / 0.5
                                )
                                best = max(best, 4 / max(gen + pause, train))
                self.assertGreaterEqual(r["sequences_per_second"] + 1e-12, best)
                self.assertLess(r["sequences_per_second"] / best, 1.02)


if __name__ == "__main__":
    unittest.main()
