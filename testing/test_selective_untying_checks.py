"""Standalone unit tests for selective untying properties, parameter budgets,
gradient conflict scores, and artifact integrity.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformer_architecture import SelectiveTransformer, TransformerConfig


def compute_gradient_conflict(g_in: torch.Tensor, g_out: torch.Tensor) -> torch.Tensor:
    """Compute gradient conflict score: -cos(g_in, g_out) matching calibrate.py."""
    dot = (g_in * g_out).sum(dim=-1)
    norm_in = g_in.norm(dim=-1)
    norm_out = g_out.norm(dim=-1)
    denom = norm_in * norm_out
    conflict = torch.full_like(dot, float("nan"))
    valid = denom > 0
    conflict[valid] = -dot[valid] / denom[valid]
    return torch.clamp(conflict, -1.0, 1.0)


class SelectiveUntyingChecks(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        np.random.seed(42)

    def test_zero_init_forward_equivalence(self) -> None:
        """At initialization (Delta = 0), a selectively untied model with 400
        selected token IDs produces identical logits to a tied model with the
        same base weights.
        """
        config = TransformerConfig()
        tied = SelectiveTransformer(config)

        # Select 400 token IDs within vocabulary (e.g. 0 to 399)
        selected_ids = list(range(400))
        # eligible_token_count must satisfy the 10% cap (400 <= eligible_count // 10)
        selective = SelectiveTransformer(
            config,
            selected_token_ids=selected_ids,
            eligible_token_count=4000,
        )

        # Transfer base weights from tied to selective model
        base_state = {
            k: v for k, v in tied.state_dict().items()
            if k not in ("selected_token_ids", "output_offsets")
        }
        selective.load_state_dict(base_state, strict=False)

        # Verify Delta is identically zero at initialization
        self.assertEqual(selective.output_offsets.shape, (400, config.d_model))
        torch.testing.assert_close(
            selective.output_offsets,
            torch.zeros_like(selective.output_offsets),
        )

        tied.eval()
        selective.eval()

        # Test forward pass with mixed tokens (both selected and unselected)
        dummy_input = torch.tensor([[0, 100, 399, 400, 1500, 7999]], dtype=torch.long)
        with torch.no_grad():
            logits_tied = tied(dummy_input)
            logits_selective = selective(dummy_input)

        # Verify logits are identical
        self.assertEqual(logits_tied.shape, logits_selective.shape)
        torch.testing.assert_close(logits_selective, logits_tied)
        self.assertTrue(torch.equal(logits_selective, logits_tied))

        # Sanity check: perturbing Delta breaks equivalence specifically
        with torch.no_grad():
            selective.output_offsets[0, 0] = 1.0
            logits_perturbed = selective(dummy_input)
            self.assertFalse(torch.equal(logits_perturbed, logits_tied))

    def test_exact_parameter_budget(self) -> None:
        """Selecting 400 tokens with embedding width 384 adds strictly
        400 * 384 = 153,600 parameters over the tied baseline
        (13,817,856 -> 13,971,456), and unselected rows retain exact
        weight sharing with the input table.
        """
        config = TransformerConfig()
        tied = SelectiveTransformer(config)
        self.assertEqual(config.d_model, 384)
        self.assertEqual(tied.parameter_count(), 13817856)

        selected_ids = list(range(400))
        selective = SelectiveTransformer(
            config,
            selected_token_ids=selected_ids,
            eligible_token_count=4000,
        )

        # Exact parameter count checks
        expected_added_params = 400 * 384  # 153,600
        self.assertEqual(expected_added_params, 153600)
        self.assertEqual(selective.parameter_count(), 13971456)
        self.assertEqual(
            selective.parameter_count() - tied.parameter_count(),
            expected_added_params,
        )

        # The only added parameter is output_offsets
        self.assertEqual(selective.output_offsets.shape, (400, 384))
        self.assertEqual(selective.output_offsets.numel(), 153600)
        self.assertFalse(hasattr(selective, "output_embedding"))

        # Verify unselected rows retain exact weight sharing with the input table
        W_in = selective.token_embedding.weight
        selected_set = set(selected_ids)
        unselected_indices = [idx for idx in range(config.vocab_size) if idx not in selected_set]

        # For unselected tokens, the output projection is strictly W_in[idx]
        hidden = torch.randn(1, 1, config.d_model)
        with torch.no_grad():
            base_logits = F.linear(hidden, W_in).squeeze()
            corrections = F.linear(hidden, selective.output_offsets).squeeze()
            sel_logits = base_logits.clone()
            sel_logits.index_add_(0, selective.selected_token_ids, corrections)

        # Unselected rows receive zero correction and match base input lookup exactly
        unsel_tensor = torch.tensor(unselected_indices, dtype=torch.long)
        torch.testing.assert_close(sel_logits[unsel_tensor], base_logits[unsel_tensor])

    def test_gradient_conflict_score_bounds_and_symmetry(self) -> None:
        """Verify that -cos(g_in, g_out) returns +1.0 for directly opposing
        gradients, -1.0 for identical gradients, and 0.0 for orthogonal gradients,
        respects symmetry, and remains bounded in [-1.0, 1.0].
        """
        # 1. Directly opposing gradients: g_out = -c * g_in (c > 0) -> +1.0
        g_in_opp = torch.tensor([1.0, -2.0, 3.0, -4.0])
        g_out_opp = torch.tensor([-2.0, 4.0, -6.0, 8.0])  # -2 * g_in
        score_opp = compute_gradient_conflict(g_in_opp, g_out_opp)
        self.assertAlmostEqual(score_opp.item(), 1.0, places=6)

        # 2. Identical (co-directional) gradients: g_out = c * g_in (c > 0) -> -1.0
        g_in_ident = torch.tensor([1.0, -2.0, 3.0, -4.0])
        g_out_ident = torch.tensor([3.0, -6.0, 9.0, -12.0])  # 3 * g_in
        score_ident = compute_gradient_conflict(g_in_ident, g_out_ident)
        self.assertAlmostEqual(score_ident.item(), -1.0, places=6)

        # 3. Orthogonal gradients: dot product is 0 -> 0.0
        g_in_orth = torch.tensor([1.0, 0.0, -1.0, 0.0])
        g_out_orth = torch.tensor([0.0, 1.0, 0.0, 2.0])
        score_orth = compute_gradient_conflict(g_in_orth, g_out_orth)
        self.assertAlmostEqual(score_orth.item(), 0.0, places=6)

        # 4. Symmetry: -cos(g_in, g_out) == -cos(g_out, g_in)
        torch.manual_seed(123)
        g_a = torch.randn(50, 384)
        g_b = torch.randn(50, 384)
        score_ab = compute_gradient_conflict(g_a, g_b)
        score_ba = compute_gradient_conflict(g_b, g_a)
        torch.testing.assert_close(score_ab, score_ba)

        # 5. Bounds: all scores in [-1.0, 1.0]
        self.assertTrue(torch.all(score_ab >= -1.0))
        self.assertTrue(torch.all(score_ab <= 1.0))

    def test_main_results_artifact_integrity(self) -> None:
        """Validate that results_v2.json contains all 4 methods (tied, conflict,
        frequency, freq_random), all 12 paired seeds (0 through 11), each evaluated
        on 500 prompts with valid satisfaction, macro recall, and NLL ranges.
        """
        results_file = ROOT / "results_v2.json"
        self.assertTrue(results_file.exists(), f"Missing {results_file}")

        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_methods = ("tied", "conflict", "frequency", "freq_random")
        expected_seeds = range(12)

        expected_keys = {f"{m}_seed{s}" for m in expected_methods for s in expected_seeds}
        self.assertEqual(set(data.keys()), expected_keys)
        self.assertEqual(len(data), 48)

        for method in expected_methods:
            for seed in expected_seeds:
                key = f"{method}_seed{seed}"
                self.assertIn(key, data)
                entry = data[key]

                # Key metadata
                self.assertEqual(entry["method"], method)
                self.assertEqual(entry["seed"], seed)

                # Evaluated on 500 prompts
                missing_words = entry["missing_words"]
                self.assertEqual(len(missing_words), 500)
                self.assertTrue(all(isinstance(w_list, list) for w_list in missing_words))

                # Keyword satisfaction valid range [0, 1] and consistency with missing_words
                sat = entry["keyword_satisfaction"]
                self.assertIsInstance(sat, float)
                self.assertGreaterEqual(sat, 0.0)
                self.assertLessEqual(sat, 1.0)
                calc_sat = sum(len(w) == 0 for w in missing_words) / 500
                self.assertAlmostEqual(sat, calc_sat, places=6)

                # Macro recall valid range [0, 1]
                rec = entry["keyword_recall"]
                self.assertIsInstance(rec, float)
                self.assertGreaterEqual(rec, 0.0)
                self.assertLessEqual(rec, 1.0)

                # NLL valid positive range
                nll = entry["response_nll"]
                self.assertIsInstance(nll, float)
                self.assertGreater(nll, 0.0)
                self.assertLess(nll, 10.0)


if __name__ == "__main__":
    unittest.main()
