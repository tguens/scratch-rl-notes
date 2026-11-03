"""Independent checks of the finite-batch timing simulation (NumPy optional)."""

import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "Timing preview tests require NumPy")
class BatchTimingTests(unittest.TestCase):
    def test_event_sum_matches_token_by_token_simulation(self):
        from utils.make_batch_timing_preview import decode_trace

        streams = [[2, 3, 1, 5, 2], [3, 3, 4, 2, 1], [1, 4, 2, 3, 3]]
        horizon, prompt, constant, kv_seconds = 12, 7, 0.4, 0.03
        trace = decode_trace(streams, horizon, prompt, constant, kv_seconds)
        indices, ages = [0] * 3, [0] * 3
        clock, peak, times, lengths = 0.0, 0, [], []
        for _ in range(horizon):
            ages = [age + 1 for age in ages]
            cache = 3 * prompt + sum(ages)
            peak = max(peak, cache)
            clock += constant + kv_seconds * cache
            for g in range(3):
                if ages[g] == streams[g][indices[g]]:
                    times.append(clock)
                    lengths.append(ages[g])
                    indices[g] += 1
                    ages[g] = 0
        np.testing.assert_allclose(trace["times"], times, rtol=1e-13)
        np.testing.assert_array_equal(trace["lengths"], lengths)
        self.assertAlmostEqual(trace["end_time"], clock)
        self.assertEqual(trace["peak_positions"], peak)

    def test_pooling_keeps_surplus_completions_and_excludes_startup(self):
        from utils.make_batch_timing_preview import batch_samples, decode_trace

        a = decode_trace([[2] * 6], 12, 0, 1.0, 0.0)
        b = decode_trace([[3] * 4], 12, 0, 1.0, 0.0)
        intervals, lengths = batch_samples([a, b], n_sequences=2, warmup=1, batches=2)
        np.testing.assert_array_equal(intervals, [3, 2])
        np.testing.assert_array_equal(lengths, [[2, 2], [3, 2]])
        with self.assertRaises(ValueError):
            batch_samples([a, b], n_sequences=2, warmup=1, batches=5)

    def test_trainer_uses_squared_lengths_and_causal_diagonal(self):
        from allocation import HARDWARE, MODELS, _model_parameters
        from utils.make_batch_timing_preview import LAYOUT, trainer_times

        model, hw = MODELS[LAYOUT["model"]], HARDWARE[LAYOUT["hardware"]]
        layout = dict(LAYOUT, prompt_tokens=2, trainer_tp=2, epochs=3)
        lengths = np.array([[1, 5], [3, 3]])  # Same tokens; different attention work.
        got = trainer_times(lengths, model, hw, layout)
        expected = []
        for response in lengths:
            ell = response + 2
            tokens = int(ell.sum())
            squares = int((ell * ell).sum())
            work = (
                8 * _model_parameters(model) * tokens
                + 9 * model["L"] * model["NQ"] * model["d"] * (squares + tokens)
            ) / hw["C"]
            comm = 24 * model["L"] * model["D"] * tokens / hw["W_ici"]
            expected.append(3 * (work + comm) / layout["trainer_gpus"])
        np.testing.assert_allclose(got, expected)
        self.assertGreater(got[0], got[1])

    def test_empty_completion_trace_still_integrates_cache_growth(self):
        from utils.make_batch_timing_preview import decode_trace

        trace = decode_trace([[10], [10]], 3, 2, 1.0, 0.5)
        self.assertEqual(len(trace["times"]), 0)
        self.assertAlmostEqual(trace["end_time"], 3 + 0.5 * (6 + 8 + 10))
        self.assertEqual(trace["peak_positions"], 10)


if __name__ == "__main__":
    unittest.main()
