import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_environment.py'


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'The documented readiness command must exist')
        spec = importlib.util.spec_from_file_location('check_environment', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_missing_data_reports_paths_and_preparation_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors = self.module.check_data('mind-small', Path(tmp), Path(tmp) / 'glove.txt')
            message = '\n'.join(errors)
            self.assertIn('train/behaviors.tsv', message)
            self.assertIn('prepare_data.py mind-small', message)
            self.assertIn('glove.txt', message)

    def test_adressa_accepts_empty_entities_and_needs_no_glove(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val', 'test'):
                folder = root / split
                folder.mkdir()
                (folder / 'news.tsv').write_text('a\tnews\tnews\ttitle\t\t\t[]\t[]\n')
                (folder / 'behaviors.tsv').write_text('1\tu\tt\ta\tb-1 a-0\n')
                (folder / 'entity_embedding.vec').touch()
            self.assertEqual(self.module.check_data('adressa-1week', root, root / 'absent'), [])
            (root / 'val/behaviors.tsv').write_text('')
            self.assertIn('val/behaviors.tsv', '\n'.join(self.module.check_data('adressa-1week', root, root / 'absent')))

    def test_cpu_only_torch_gives_actionable_cuda_error(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            errors = self.module.check_cuda(torch)
        self.assertIn('CUDA', '\n'.join(errors))
        self.assertIn('CPU', '\n'.join(errors))

    def test_multiple_visible_gpus_requests_single_gpu(self):
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=2):
            errors = self.module.check_cuda(torch)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', '\n'.join(errors))
