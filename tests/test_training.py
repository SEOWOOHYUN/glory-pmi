import unittest
from itertools import repeat
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch.cuda.amp import GradScaler
from main import train
from utils.common import EarlyStopping


class TrainingTests(unittest.TestCase):
    def test_infinite_loader_is_bounded_and_partial_accumulation_flushes(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.calls = 0

            def forward(self, *args):
                self.calls += 1
                return self.weight.square(), torch.tensor([[1., 0.]])

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        cfg = NS(max_steps=3, accumulation_steps=2, log_steps=100,
                 val_skip_epochs=2, dataset=NS(pos_count=100), batch_size=16,
                 val_steps=1000, num_epochs=5)
        batch = tuple(torch.tensor([0]) for _ in range(6))
        with patch('main.val', return_value={'auc': .5}) as validation:
            train(model, optimizer, GradScaler(enabled=False), scheduler,
                  repeat(batch), 'cpu', cfg, lambda _: (False, False))
        self.assertEqual(model.calls, 3)
        self.assertEqual(validation.call_count, 1)
        # First update uses two gradients at w=1: w=.6; final single gradient -> .48.
        self.assertAlmostEqual(model.weight.item(), .48, places=6)
        self.assertEqual(scheduler.last_epoch, 2)

    def test_zero_auc_still_selects_first_checkpoint(self):
        self.assertEqual(EarlyStopping(5)(0.), (False, True))
