import random
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from utils.resume import save_resume, restore_resume

class ResumeTest(unittest.TestCase):
    def test_next_update_matches_uninterrupted(self):
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.3))
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, 1, .9)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        def step():
            opt.zero_grad()
            model(torch.randn(4, 3)).sum().backward()
            opt.step(); sched.step()
        step()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'last.pt'
            save_resume(path, model, opt, sched, scaler, {'step': 1}, {'data': 'a'})
            step()
            expected = [p.detach().clone() for p in model.parameters()]
            state = restore_resume(path, model, opt, sched, scaler, {'data': 'a'})
            self.assertEqual(state['step'], 1)
            step()
            for a,b in zip(expected,model.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            with self.assertRaises(ValueError):
                restore_resume(path,model,opt,sched,scaler,{'data':'other'})

if __name__ == '__main__': unittest.main()
