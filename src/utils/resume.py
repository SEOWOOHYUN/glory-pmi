"""Atomic full-state checkpoints; only load checkpoints from a trusted run."""
import os
import random
from pathlib import Path
import numpy as np
import torch


def save_resume(path, model, optimizer, scheduler, scaler, progress, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=1, signature=signature, progress=progress,
                   model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                   python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def restore_resume(path, model, optimizer, scheduler, scaler, signature):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('version') != 1 or payload.get('signature') != signature:
        raise ValueError('Resume checkpoint configuration/data mismatch')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    random.setstate(payload['python_rng'])
    np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng']:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['progress']
