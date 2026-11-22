import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('run_experiment', Path(__file__).resolve().parents[1] / 'scripts/run_experiment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RunnerTests(unittest.TestCase):
    def test_datasets_share_training_conditions(self):
        args = module.parser().parse_args(['--dataset', 'mind-small', '--method', 'count'])
        first = module.build_command(args, Path('/tmp/data'), Path('/tmp/result'), 100, 20, 20)
        args.dataset = 'adressa-1week'
        second = module.build_command(args, Path('/tmp/data'), Path('/tmp/result'), 100, 20, 20)
        a, b = set(first), set(second)
        self.assertEqual(a - b, {'small'})
        self.assertEqual(b - a, {'adressa'})
        for setting in ['batch_size=16', 'accumulation_steps=2', 'optimizer.lr=0.0002',
                        'model.num_neighbors=8', 'model.k_hops=2', 'seed=3407']:
            self.assertIn(setting, a)

    def test_methods_only_change_scoring(self):
        args = module.parser().parse_args(['--dataset', 'mind-small', '--method', 'count'])
        a = module.build_command(args, Path('/tmp/data'), Path('/tmp/result'), 100, 20, 20)
        args.method = 'ppmi'
        b = module.build_command(args, Path('/tmp/data'), Path('/tmp/result'), 100, 20, 20)
        self.assertEqual(set(a) - set(b), {'neighbor_method=count'})
        self.assertEqual(set(b) - set(a), {'neighbor_method=ppmi'})


if __name__ == '__main__':
    unittest.main()

class CacheTests(unittest.TestCase):
    def test_changed_data_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / 'train').mkdir()
            (data / 'train/news_dict.bin').write_bytes(b'old')
            with self.assertRaises(ValueError):
                module.check_cache(data, {'train/news.tsv': 'new'}, False)
            module.check_cache(data, {'train/news.tsv': 'new'}, True)
            (data / 'experiment-cache.json').write_text('{"train/news.tsv": "new"}')
            module.check_cache(data, {'train/news.tsv': 'new'}, False)
            with self.assertRaises(ValueError):
                module.check_cache(data, {'train/news.tsv': 'changed'}, False)
