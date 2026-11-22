"""Check prepared inputs and the single-GPU runtime without starting training."""
import argparse
import importlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_data(dataset, data, glove):
    errors = []
    for split in ('train', 'val', 'test'):
        for name in ('news.tsv', 'behaviors.tsv', 'entity_embedding.vec'):
            path = data / split / name
            allow_empty = dataset == 'adressa-1week' and name == 'entity_embedding.vec'
            try:
                with path.open('rb') as stream:
                    if not stream.read(1) and not allow_empty:
                        errors.append(f'Empty data file: {path}')
            except OSError as exc:
                errors.append(f'Cannot read {path}: {exc}')
    if errors:
        errors.append(f'Prepare data first: python scripts/prepare_data.py {dataset} --help')
    if dataset == 'mind-small':
        try:
            with glove.open('rb') as stream:
                if not stream.read(1):
                    errors.append(f'Empty GloVe file: {glove}')
        except OSError:
            errors.append(f'MIND needs glove.840B.300d.txt at {glove}; see README data preparation')
    return errors


def check_cuda(torch):
    if not torch.cuda.is_available():
        return ['CUDA is unavailable: use a Linux NVIDIA GPU runtime and a CUDA torch wheel; '
                'CPU/Mac checks can use --cpu-only (no training).']
    if torch.cuda.device_count() != 1:
        return ['Expose exactly one GPU: export CUDA_VISIBLE_DEVICES=0 before running Python.']
    if not torch.distributed.is_nccl_available():
        return ['NCCL is unavailable; training needs a Linux CUDA PyTorch build.']
    try:
        sample = torch.ones((2, 2), device='cuda')
        (sample @ sample).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return [f'CUDA tensor operation failed: {exc}']
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['mind-small', 'adressa-1week'])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--cpu-only', action='store_true', help='skip GPU checks; does not enable CPU training')
    args = parser.parse_args()
    folder = 'MINDsmall' if args.dataset == 'mind-small' else 'AdressaSmall'
    data = (args.data_dir or ROOT / 'data' / folder).resolve()
    errors = check_data(args.dataset, data, ROOT / 'data/glove/glove.840B.300d.txt')
    modules = {}
    for name in ('torch', 'torch_geometric', 'numpy', 'pandas', 'nltk', 'sklearn',
                 'pyrootutils', 'tqdm', 'omegaconf', 'wandb', 'hydra'):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:
            errors.append(f'Cannot import {name}: {exc}; install requirements.txt in the active environment')
    torch = modules.get('torch')
    if torch is not None and not args.cpu_only:
        errors.extend(check_cuda(torch))
    result = {'ok': not errors, 'mode': 'CPU checks only' if args.cpu_only else 'CUDA readiness',
              'python': platform.python_version(), 'data_dir': str(data),
              'torch': torch.__version__ if torch is not None else None,
              'torch_cuda': torch.version.cuda if torch is not None else None,
              'errors': errors,
              'scope': 'File presence/imports/runtime only; not full data validation or a training benchmark.'}
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
