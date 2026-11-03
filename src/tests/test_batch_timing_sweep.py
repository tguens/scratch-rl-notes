"""Memory and rank checks for the multi-layout timing preview."""

import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "Timing sweep checks require NumPy")
class TimingSweepTests(unittest.TestCase):
    def test_sampler_batches_fit_even_at_the_response_cap(self):
        from allocation import HARDWARE, MODELS, _model_parameters
        from utils.make_batch_timing_sweep import sampler_batches

        model, hardware = MODELS["Llama-3.1-8B"], HARDWARE["H100"]
        cap, prompt = 30720, 512
        kappa = 4 * model["L"] * model["NKV"] * model["d"]
        parameters = _model_parameters(model)
        for tp in (1, 2, 4, 8):
            choices = sampler_batches(model, hardware, cap, tp)
            self.assertEqual(len(choices), 2)
            self.assertLess(choices[0][1], choices[1][1])
            for _, batch in choices:
                memory = (2 * parameters + kappa * batch * (prompt + cap)) / tp
                self.assertLessEqual(memory, 0.9 * hardware["HBM"])
            next_batch = choices[-1][1] + 1
            self.assertGreater(
                (2 * parameters + kappa * next_batch * (prompt + cap)) / tp,
                0.9 * hardware["HBM"],
            )

    def test_trainer_microbatches_fit_and_divide_the_global_batch(self):
        from allocation import HARDWARE, MODELS, _activation_bytes, _model_parameters
        from utils.make_batch_timing_sweep import trainer_microbatches

        model, hardware = MODELS["Llama-3.1-8B"], HARDWARE["H100"]
        for n in (8, 16, 32, 64, 128):
            for tp in (1, 2, 4, 8):
                ranks = n // tp
                for nseq in (256, 1024, 4096):
                    sizes = trainer_microbatches(model, hardware, 30720, n, tp, nseq)
                    self.assertIn(1, sizes)
                    for size in sizes:
                        self.assertEqual(nseq % (ranks * size), 0)
                        memory = (
                            16 * _model_parameters(model) / n
                            + size * _activation_bytes(model, 31232) / tp
                        )
                        self.assertLessEqual(memory, 0.9 * hardware["HBM"])
        self.assertEqual(trainer_microbatches(model, hardware, 30720, 128, 1, 64), [])

    def test_summary_preserves_run_ranges_and_counts(self):
        from utils.make_batch_timing_sweep import summarize

        result = summarize([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])])
        self.assertEqual(result["samples"], 5)
        self.assertEqual(result["mean_seconds"], 3.0)
        self.assertAlmostEqual(result["p99_seconds"], 4.96)
        self.assertEqual(result["run_mean_min"], 2.0)
        self.assertEqual(result["run_mean_max"], 4.5)


if __name__ == "__main__":
    unittest.main()
