"""One entry point and shared training conditions for both datasets and all arms."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dataload.neighbor_scoring import read_training, head_items


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['mind-small', 'adressa-1week'])
    default_file = ROOT / 'experiment.json'
    default_method = json.loads(default_file.read_text())['method'] if default_file.exists() else 'count'
    p.add_argument('--method', choices=['count', 'ppmi', 'shrunk_ppmi'], default=default_method)
    p.add_argument('--lambda', dest='shrinkage', type=float, default=5)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--prepare', action='store_true', help='rebuild preprocessing (first run required)')
    p.add_argument('--checkpoint', type=Path, help='evaluate this checkpoint instead of training')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-steps', type=positive_int,
                   help='cap training minibatches for a trial; final validation still runs')
    return p


def build_command(args, data, output, positives, val_len, test_len):
    config = 'small' if args.dataset == 'mind-small' else 'adressa'
    return [sys.executable, str(ROOT / 'src/main.py'), '--config-name', config,
            f'dataset.dataset_dir={data}', f'result_dir={output}',
            f'path.ckp_dir={output / "checkpoints"}', f'path.pred_dir={output}',
            f'neighbor_method={args.method}', f'pmi_lambda={args.shrinkage}',
            f'seed={args.seed}', 'batch_size=16', 'accumulation_steps=2',
            'optimizer.lr=0.0002', 'num_epochs=5', 'gpu_num=1', 'npratio=4',
            'model.num_neighbors=8', 'model.k_hops=2', 'model.directed=true',
            'model.use_graph_type=0', 'model.head_num=20', 'model.head_dim=20',
            'dropout_probability=0.2', 'warmup_ratio=0.1', 'early_stop_patience=5',
            'val_skip_epochs=2', 'val_steps=1000', 'num_workers=0',
            f'dataset.pos_count={positives}', f'dataset.val_len={val_len}',
            f'dataset.test_len={test_len}', 'ml_label=experiment',
            f'reprocess={str(args.prepare).lower()}', 'export_predictions=true',
            f'eval_only={str(args.checkpoint is not None).lower()}', f'eval_split={args.split}',
            f'max_steps={args.max_steps if args.max_steps is not None else "null"}',
            f'checkpoint_path={args.checkpoint.resolve() if args.checkpoint else "null"}']


def count_rows(path, training=False):
    count = 0
    with path.open() as stream:
        for line in stream:
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) != 5:
                raise ValueError(f'Expected five behavior fields: {path}')
            if training:
                labels = [token.rsplit('-', 1)[1] for token in fields[4].split()]
                count += labels.count('1') if '0' in labels else 0
            else:
                count += bool(fields[3].strip())
    return count


def check_cache(data, fingerprint, rebuild):
    stamp = data / 'experiment-cache.json'
    cached = (data / 'train/news_dict.bin').exists()
    if cached and not rebuild and (not stamp.exists() or json.loads(stamp.read_text()) != fingerprint):
        raise ValueError('Preprocessing cache is unverified or data changed; rerun with --prepare')


def main():
    args = parser().parse_args()
    if args.shrinkage not in (1, 5, 20, 100):
        raise ValueError('Prespecified lambda grid: 1, 5, 20, 100')
    if args.split == 'test' and not args.checkpoint:
        raise ValueError('Test is evaluation-only; first select a checkpoint on validation')
    if args.dataset == 'mind-small' and args.split == 'test':
        raise ValueError('MIND-small has no independent test in this protocol; use val')
    folder = 'MINDsmall' if args.dataset == 'mind-small' else 'AdressaSmall'
    data = (args.data_dir or ROOT / 'data' / folder).resolve()
    run_name = f'{args.dataset}-{args.method}-lambda{args.shrinkage:g}-seed{args.seed}'
    if args.checkpoint:
        run_name += f'-eval-{args.split}'
    elif args.max_steps is not None:
        run_name += f'-trial{args.max_steps}'
    output = (args.output or ROOT / 'runs' / run_name).resolve()
    counts = [count_rows(data / split / 'behaviors.tsv', split == 'train') for split in ('train', 'val', 'test')]
    if not all(counts):
        raise ValueError('All prepared splits must be nonempty')
    command = build_command(args, data, output, *counts)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite a run: {output}; choose --output')
    output.mkdir(parents=True)
    _, popularity, info = read_training(data / 'train/behaviors.tsv')
    (output / 'popularity.json').write_text(json.dumps({'popularity': popularity, 'head': sorted(head_items(popularity))}))
    files = {}
    for split in ('train', 'val', 'test'):
        for name in ('news.tsv', 'behaviors.tsv', 'entity_embedding.vec'):
            digest = hashlib.sha256()
            with (data / split / name).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            files[f'{split}/{name}'] = digest.hexdigest()
    check_cache(data, files, args.prepare)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    command.append(f'data_fingerprint={fingerprint}')
    manifest = {'dataset': args.dataset, 'method': args.method, 'lambda': args.shrinkage,
                'seed': args.seed, 'command': command, 'training': info, 'files_sha256': files,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                'scope': 'validation' if args.split == 'val' else 'held-out test',
                'accumulation': 'upstream sum of two minibatch mean losses, no division by 2'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    environment = dict(os.environ, PROJECT_ROOT=str(ROOT), WANDB_MODE='disabled')
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    (data / 'experiment-cache.json').write_text(json.dumps(files, indent=2) + '\n')


if __name__ == '__main__':
    main()
