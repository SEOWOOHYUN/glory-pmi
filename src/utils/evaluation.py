import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from dataload.data_load import load_data
from dataload.neighbor_scoring import read_training, head_items
from utils.experiment_metrics import ranking_metrics


def evaluate(model, device, cfg, split='val'):
    model.eval()
    loader = load_data(cfg, mode=split, model=model, local_rank=device)
    output = Path(cfg.result_dir or cfg.path.pred_dir)
    output.mkdir(parents=True, exist_ok=True)
    pop_file = output / 'popularity.json'
    if pop_file.exists():
        head = set(json.loads(pop_file.read_text())['head'])
    else:
        _, popularity, _ = read_training(Path(cfg.dataset.train_dir) / 'behaviors.tsv')
        head = head_items(popularity)
    all_metrics = []
    prediction_path = output / f'{split}-latest.jsonl'
    with prediction_path.open('w') as stream, torch.no_grad():
        for batch in tqdm(loader, desc=f'Evaluating {split}'):
            graph, mappings, clicked, candidates, entity, mask, labels, metadata = batch
            scores = model.module.validation_process(
                graph, mappings, clicked.to(device),
                torch.as_tensor(np.asarray(candidates), dtype=torch.float32, device=device),
                entity.to(device), mask.to(device))
            labels = labels.tolist()
            metrics = ranking_metrics(metadata['candidates'], labels, scores, head)
            all_metrics.append(metrics)
            if cfg.export_predictions:
                row = dict(metadata, labels=labels, scores=scores, **metrics)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    if not all_metrics:
        raise ValueError('No nonempty-history evaluation impressions')
    summary = {'evaluated_impressions': len(all_metrics)}
    for key in ('auc', 'mrr', 'ndcg5', 'ndcg10', 'head_ratio10', 'candidate_head_ratio', 'head_amplification10'):
        values = [row[key] for row in all_metrics if row[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
        summary[f'{key}_valid_count'] = len(values)
    long_rows = [row for row in all_metrics if row['candidate_count'] > 10]
    summary['candidates_gt10_count'] = len(long_rows)
    summary['head_amplification10_candidates_gt10'] = (
        sum(row['head_amplification10'] for row in long_rows) / len(long_rows) if long_rows else None)
    (output / f'{split}-latest.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary
