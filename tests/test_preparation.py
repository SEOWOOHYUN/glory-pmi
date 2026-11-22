import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('prepare_data', Path(__file__).resolve().parents[1] / 'scripts/prepare_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def test_adressa_temporal_candidates(self):
        # day 1 supplies histories and a negative pool; days 5/6/7 are targets.
        events = [('u', 'a', 1483228800), ('v', 'b', 1483228801),
                  ('w', 'c', 1483228802), ('u', 'd', 1483574400),
                  ('u', 'e', 1483660800), ('u', 'f', 1483747200)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'prepared'
            module.write_adressa(events, {n: n for n in 'abcdef'}, out, 3407, 1)
            train = (out / 'train/behaviors.tsv').read_text().strip().split('\t')
            val = (out / 'val/behaviors.tsv').read_text().strip().split('\t')
            test = (out / 'test/behaviors.tsv').read_text().strip().split('\t')
            self.assertEqual(train[3], 'N1')
            self.assertEqual(val[3], 'N1 N4')
            self.assertEqual(test[3], 'N1 N4 N5')
            self.assertNotIn('N5', train[4])
            self.assertNotIn('N6', val[4])
            self.assertEqual(len(val[4].split()), 2)
            self.assertEqual(len((out / 'train/news.tsv').read_text().splitlines()), 4)
            self.assertEqual(len((out / 'val/news.tsv').read_text().splitlines()), 5)
            with self.assertRaises(FileExistsError):
                module.write_adressa(events, {n: n for n in 'abcdef'}, out, 3407, 1)


if __name__ == '__main__':
    unittest.main()

class SimultaneousTests(unittest.TestCase):
    def test_simultaneous_clicks_are_not_each_others_history(self):
        events = [('u', 'a', 1483228800), ('v', 'b', 1483228801),
                  ('u', 'c', 1483574400), ('u', 'd', 1483574400)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'prepared'
            module.write_adressa(events, {n: n for n in 'abcd'}, out, negatives=1)
            rows = [row.split('\t') for row in (out / 'train/behaviors.tsv').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual([row[3] for row in rows], ['N1', 'N1'])
