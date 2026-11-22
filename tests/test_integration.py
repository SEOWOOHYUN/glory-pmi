import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from torch_geometric.data import Data

from dataload.dataset import ValidGraphDataset
from dataload.data_preprocess import prepare_neighbor_list
from dataload.neighbor_scoring import neighbor_cache_name
import pickle


class IntegrationTests(unittest.TestCase):
    def test_empty_history_does_not_repeat_or_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'b.tsv'
            path.write_text('0\tu\tt\t\ta-1\n1\tu\tt\ta\ta-1\n2\tu\tt\t\ta-1\n')
            obj = object.__new__(ValidGraphDataset)
            obj.filename = path
            obj.line_mapper = lambda line: tuple([line.split('\t')[0]] * 7)
            self.assertEqual([row[0] for row in obj], ['1'])

    def test_method_caches_and_raw_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            edges = torch.tensor([[1, 1, 2, 3], [4, 5, 4, 5]])
            counts = torch.tensor([20, 80, 5, 95])
            torch.save(Data(edge_index=edges, edge_attr=counts, num_nodes=6), folder / 'nltk_news_graph.pt')
            with open(folder / 'news_dict.bin', 'wb') as f:
                pickle.dump({str(i): i for i in range(1, 6)}, f)
            cfg = NS(dataset=NS(train_dir=tmp, val_dir=tmp, test_dir=tmp),
                     model=NS(directed=True, use_graph_type=0), reprocess=False,
                     reprocess_neighbors=False, neighbor_method='count', pmi_lambda=5.)
            for method, expected in [('count', [1, 2]), ('ppmi', [2, 1])]:
                cfg.neighbor_method = method
                prepare_neighbor_list(cfg)
                with open(folder / neighbor_cache_name(cfg), 'rb') as f:
                    neighbors = pickle.load(f)
                self.assertEqual(neighbors[4], expected)
                self.assertEqual(neighbors[0], [])
            graph = torch.load(folder / 'nltk_news_graph.pt', weights_only=False)
            self.assertTrue(torch.equal(graph.edge_attr, counts))


if __name__ == '__main__':
    unittest.main()
