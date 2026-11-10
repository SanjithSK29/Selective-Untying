"""Correctness checks requiring no dataset, download, or original checkpoint."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from calibrate import compute_grads
from common import load_pretrained
from reproducibility import prepare_results, atomic_json
from sft import collate, evaluate
from transformer_architecture import SelectiveTransformer, TransformerConfig


class Correctness(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.config = TransformerConfig(vocab_size=20, context_length=16,
                                        n_layers=1, d_model=8, n_heads=2, d_ff=16)
        self.x = torch.tensor([[2, 3, 2, 4]])
        self.y = torch.tensor([[3, 2, 4, 1]])

    def test_checkpoint_modes_and_reject_corruption(self):
        base = SelectiveTransformer(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            state = base.state_dict()
            torch.save({'model': state}, path)
            for mode in ('tied', 'full', 'selective'):
                kw = dict(selected_token_ids=[2], eligible_token_count=20) if mode == 'selective' else {}
                model = SelectiveTransformer(self.config, output_mode=mode, **kw)
                load_pretrained(model, path, torch.device('cpu'))
                torch.testing.assert_close(model(self.x), base(self.x))
            del state['final_norm.weight']
            torch.save({'model': state}, path)
            with self.assertRaisesRegex(ValueError, 'keys mismatch'):
                load_pretrained(base, path, torch.device('cpu'))

    def test_decomposition_matches_independent_untied_gradients(self):
        tied = SelectiveTransformer(self.config)
        gi, go = compute_grads(tied, self.x, self.y)
        full = SelectiveTransformer(self.config, output_mode='full')
        full.load_state_dict(tied.state_dict(), strict=False)
        full.init_output_from_input()
        torch.nn.functional.cross_entropy(full(self.x).flatten(0, 1), self.y.flatten()).backward()
        torch.testing.assert_close(gi, full.token_embedding.weight.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(go, full.output_embedding.weight.grad)

    def test_masking_and_per_example_metrics(self):
        class Tok:
            def token_to_id(self, value): return 1
            def decode(self, ids): return 'cat' if ids else ''
        pairs = [([2, 3], [4, 1], ['cat']), ([2], [5, 6, 1], ['dog'])]
        inputs, targets = collate(pairs, Tok(), 'cpu')
        self.assertEqual(targets.tolist(), [[-100, 4, 1, -100], [5, 6, 1, -100]])
        model = SelectiveTransformer(self.config)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'examples.json'
            with patch('sft.generate', side_effect=[[2, 3, 4, 1], [2, 5, 1]]):
                sat, nll = evaluate(model, pairs, Tok(), 16, 'cpu', out)
            import json
            records = json.loads(out.read_text())
            self.assertEqual(sat, .5)
            self.assertEqual([r['missing_words'] for r in records], [[], ['dog']])
            self.assertEqual([r['keyword_recall'] for r in records], [1., 0.])
            self.assertAlmostEqual(nll, sum(r['response_nll_sum'] for r in records) / 5, places=6)

    def test_resume_rejects_legacy_and_changed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            atomic_json(path, {})
            with self.assertRaises(ValueError): prepare_results(path, {'a': 1})
            path.unlink()
            self.assertEqual(prepare_results(path, {'a': 1}), {})
            atomic_json(path, {'run': {}})
            with self.assertRaises(ValueError): prepare_results(path, {'a': 2})
            self.assertIn('run', prepare_results(path, {'a': 1}))

    def test_data_conversion_preserves_multiline_and_last_record(self):
        import csv
        from prepare_data import to_csv
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.txt'
            target = Path(directory) / 'data.csv'
            source.write_text('First story.\nSecond line.\n<|endoftext|>Last story.', encoding='utf-8')
            self.assertEqual(to_csv(source, target), 2)
            with target.open(encoding='utf-8', newline='') as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows, [['text'], ['First story.\nSecond line.'], ['Last story.']])

    def test_cpu_sft_and_alpha_one_equivalence(self):
        import sft_gradscale
        from sft import configure_optimizer, get_lr
        tok = Tokenizer(WordLevel({'<unk>': 0, '<eos>': 1, 'cat': 2}, unk_token='<unk>'))
        pairs = [([2], [2, 1], ['cat'])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'base.pt'
            torch.save({'model': SelectiveTransformer(self.config).state_dict()}, path)
            args = argparse.Namespace(ckpt_path=str(path), weight_decay=.1, lr=1e-4,
                    warmup_steps=1, max_steps=1, min_lr=1e-5, batch_size=1,
                    grad_clip=1., artifact_dir=directory)
            with patch.object(sft_gradscale, 'SelectiveTransformer',
                              side_effect=lambda: SelectiveTransformer(self.config)), \
                 patch.object(sft_gradscale, 'evaluate', return_value=(.5, 1.)):
                sft_gradscale.run_one(args, 1., 0, pairs, pairs, tok, 16, torch.device('cpu'), {})
            actual = torch.load(Path(directory) / 'gradscale_a1_seed0.pt', weights_only=True)['model']
            torch.manual_seed(0)
            model = SelectiveTransformer(self.config)
            load_pretrained(model, path, torch.device('cpu'))
            optimizer = configure_optimizer(model, .1, 1e-4)
            inputs, targets = collate(pairs, tok, 'cpu')
            loss = torch.nn.functional.cross_entropy(model(inputs).flatten(0, 1), targets.flatten(), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(actual[key], value)


if __name__ == '__main__':
    unittest.main()
