"""Per-impression ranking and paired user-cluster uncertainty estimates."""
import math
import random
from collections import defaultdict


def ranking_metrics(items, labels, scores, head):
    if not items or len(items) != len(labels) or len(items) != len(scores):
        raise ValueError('nonempty aligned candidates, labels and scores required')
    if any(label not in (0, 1) for label in labels) or not all(math.isfinite(s) for s in scores):
        raise ValueError('binary labels and finite scores required')
    order = sorted(range(len(items)), key=lambda i: -scores[i])
    positives = sum(labels)
    negatives = len(labels) - positives
    # Tie-aware Mann-Whitney AUC in O(n log n).
    wins, neg_before, index = 0., 0, 0
    ascending = sorted(range(len(items)), key=lambda i: scores[i])
    while index < len(items):
        end = index + 1
        while end < len(items) and scores[ascending[end]] == scores[ascending[index]]:
            end += 1
        pos = sum(labels[i] for i in ascending[index:end])
        neg = end - index - pos
        wins += pos * (neg_before + neg / 2)
        neg_before += neg
        index = end
    result = {'auc': wins / (positives * negatives) if positives and negatives else None,
              'mrr': sum(labels[i] / (rank + 1) for rank, i in enumerate(order)) / positives if positives else None}
    for k in (5, 10):
        dcg = sum(labels[i] / math.log2(rank + 2) for rank, i in enumerate(order[:k]))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, positives)))
        result[f'ndcg{k}'] = dcg / ideal if ideal else None
    top = order[:10]
    candidate_head = sum(n in head for n in items) / len(items)
    result.update(candidate_count=len(items),
                  head_ratio10=sum(items[i] in head for i in top) / len(top),
                  candidate_head_ratio=candidate_head)
    result['head_amplification10'] = result['head_ratio10'] - candidate_head
    return result


def paired_bootstrap(baseline, treatment, metric, repeats=2000, seed=3407):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    def keyed(rows):
        result = {str(row['impression_id']): row for row in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate impression IDs')
        return result
    a, b = keyed(baseline), keyed(treatment)
    if a.keys() != b.keys() or not a:
        raise ValueError('paired runs require identical nonempty impression IDs')
    clusters = defaultdict(list)
    for iid, row in a.items():
        other = b[iid]
        if row['user_id'] != other['user_id']:
            raise ValueError('user mismatch')
        for field in ('candidates', 'labels'):
            if field in row or field in other:
                if row.get(field) != other.get(field):
                    raise ValueError(f'{field} mismatch')
        if (row[metric] is None) != (other[metric] is None):
            raise ValueError('metric validity differs between runs')
        if row[metric] is not None:
            delta = other[metric] - row[metric]
            if not math.isfinite(delta):
                raise ValueError('non-finite paired metric')
            clusters[row['user_id']].append(delta)
    if not clusters:
        raise ValueError('no valid paired observations')
    totals = [(sum(values), len(values)) for values in clusters.values()]
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        selected = rng.choices(totals, k=len(totals))
        samples.append(sum(s for s, n in selected) / sum(n for s, n in selected))
    samples.sort()
    return {'metric': metric, 'delta': sum(s for s, n in totals) / sum(n for s, n in totals),
            'ci_low': samples[int(.025 * (repeats - 1))],
            'ci_high': samples[int(.975 * (repeats - 1))],
            'users': len(totals), 'impressions': sum(n for s, n in totals),
            'repeats': repeats, 'seed': seed}
