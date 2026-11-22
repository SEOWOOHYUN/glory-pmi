import math
import tempfile
import unittest
from pathlib import Path

from dataload.neighbor_scoring import read_training, score_edges, rank_incoming, head_items, graph_metrics
from utils.experiment_metrics import ranking_metrics, paired_bootstrap


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.edges = {('a', 'x'): 20, ('a', 'y'): 80, ('b', 'x'): 5,
                      ('c', 'y'): 95}

    def test_directed_marginals_and_shrinkage(self):
        scores = score_edges(self.edges, 'ppmi', 0)
        self.assertAlmostEqual(scores[('a', 'x')], math.log(200 * 20 / (100 * 25)))
        self.assertEqual(scores[('a', 'y')], 0)
        corrected = score_edges(self.edges, 'shrunk_ppmi', 5)
        self.assertAlmostEqual(corrected[('b', 'x')], math.log(8) / 2)
        self.assertEqual(scores, score_edges(self.edges, 'shrunk_ppmi', 0))

    def test_incoming_order_and_fallback(self):
        self.assertEqual(rank_incoming(self.edges, 'count', 0)['x'], ['a', 'b'])
        self.assertEqual(rank_incoming(self.edges, 'ppmi', 0)['x'], ['b', 'a'])
        self.assertEqual(rank_incoming(self.edges, 'ppmi', 0)['y'], ['c', 'a'])
        self.assertEqual(rank_incoming({}, 'count', 0), {})
        self.assertEqual(self.edges[('a', 'x')], 20)
        with self.assertRaises(ValueError):
            score_edges(self.edges, 'shrunk_ppmi', -1)

    def test_train_unique_history_and_unique_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'behaviors.tsv'
            path.write_text('1\tu\tt\ta x\tx-1 y-0\n2\tu\tt\ta x\tx-1 b-1\n'
                            '3\tv\tt\t\ta-1 x-0\n4\tv\tt\tb x\tb-1\n')
            edges, pop, info = read_training(path)
        self.assertEqual(edges, {('a', 'x'): 1})
        self.assertEqual(pop, {'a': 2, 'x': 2, 'b': 2})
        self.assertEqual(info['users'], 2)
        self.assertEqual(head_items(pop), {'a'})

    def test_graph_denominators(self):
        metrics = graph_metrics(self.edges, {'a': 100, 'b': 1, 'c': 3}, 'ppmi', 0, 1)
        self.assertEqual(metrics['selected_slots'], 2)
        self.assertEqual(metrics['head_neighbor_ratio'], 0)
        self.assertEqual(metrics['replacement_rate'], 0.5)


class RankingTests(unittest.TestCase):
    def test_metrics_and_short_candidate_list(self):
        m = ranking_metrics(['a', 'b', 'c'], [1, 0, 1], [0.9, 0.8, 0.7], {'a'})
        self.assertEqual(m['auc'], 0.5)
        self.assertAlmostEqual(m['mrr'], (1 + 1/3) / 2)
        self.assertEqual(m['head_amplification10'], 0)
        self.assertEqual(m['candidate_count'], 3)
        self.assertIsNone(ranking_metrics(['a'], [1], [0.1], {'a'})['auc'])
        with self.assertRaises(ValueError):
            ranking_metrics(['a'], [1], [float('nan')], set())
        with self.assertRaises(ValueError):
            ranking_metrics(['a'], [1, 0], [0.5], set())

    def test_paired_bootstrap_requires_matched_rows(self):
        a = [{'impression_id': '1', 'user_id': 'u', 'auc': .5},
             {'impression_id': '2', 'user_id': 'v', 'auc': .6}]
        b = [dict(row, auc=row['auc'] + .1) for row in a]
        ci = paired_bootstrap(a, b, 'auc', 100, 3407)
        self.assertAlmostEqual(ci['delta'], .1)
        self.assertAlmostEqual(ci['ci_low'], .1)
        self.assertAlmostEqual(ci['ci_high'], .1)
        with self.assertRaises(ValueError):
            paired_bootstrap(a, b[:1], 'auc', 100, 3407)


if __name__ == '__main__':
    unittest.main()
