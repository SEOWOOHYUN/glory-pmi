"""Synthetic CPU checks are interface tests, never accuracy experiments."""
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch_geometric.data import Data
from models.GLORY import GLORY
from utils.common import checkpoint_signature, check_checkpoint


class ModelSmokeTests(unittest.TestCase):
    def test_both_dataset_models_forward_backward_and_evaluation(self):
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[1]
        os.environ['PROJECT_ROOT'] = str(root)
        for config in ('small', 'adressa'):
            with self.subTest(config=config), initialize_config_dir(version_base='1.2', config_dir=str(root / 'configs')):
                cfg = compose(config_name=config)
                model = GLORY(cfg, glove_emb=np.zeros((11, 300), dtype=np.float32) if config == 'small' else 10,
                              entity_emb=np.zeros((3, 100), dtype=np.float32))
                tokens = torch.zeros(4, 38, dtype=torch.long)
                tokens[:, :30] = torch.randint(1, 11, (4, 30))
                graph = Data(x=tokens, edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
                entities = torch.zeros(1, 2, 55, dtype=torch.long)
                mask = torch.zeros(1, 2, 50, dtype=torch.long)
                loss, scores = model(graph, torch.tensor([[0, 1]]), tokens[:2].unsqueeze(0),
                                     entities, mask, torch.tensor([0]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                model.eval()
                with torch.no_grad():
                    encoded = model.local_news_encoder(tokens.unsqueeze(0)).squeeze(0)
                    values = model.validation_process(Data(x=encoded, edge_index=graph.edge_index), [0, 1],
                                                      torch.zeros(2, 5, dtype=torch.long), encoded[:2],
                                                      entities[0], mask[0])
                self.assertEqual(len(values), 2)
                self.assertTrue(np.isfinite(values).all())
                checkpoint = {'experiment': checkpoint_signature(cfg)}
                check_checkpoint(checkpoint, cfg)
                cfg.neighbor_method = 'ppmi'
                with self.assertRaises(ValueError):
                    check_checkpoint(checkpoint, cfg)
