"""Training-only counts and incoming-neighbor ordering, without GPU dependencies."""
import hashlib
import math
from collections import Counter, defaultdict


METHODS = ('count', 'ppmi', 'shrunk_ppmi')


def read_training(path):
    edges, readers, seen = Counter(), defaultdict(set), set()
    rows = 0
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for raw in stream:
            digest.update(raw)
            fields = raw.decode('utf-8').rstrip('\r\n').split('\t')
            if len(fields) != 5:
                raise ValueError(f'{path}:{rows + 1}: expected 5 behavior fields')
            _, user, _, history, impressions = fields
            history = history.split()
            # Match upstream: take the FIRST row for a user, including empty history.
            if user not in seen:
                edges.update(zip(history, history[1:]))
                seen.add(user)
            clicked = set(history)
            for token in impressions.split():
                item, label = token.rsplit('-', 1)
                if label not in ('0', '1'):
                    raise ValueError(f'{path}:{rows + 1}: invalid label')
                if label == '1':
                    clicked.add(item)
            for item in clicked:
                readers[item].add(user)
            rows += 1
    return edges, {item: len(users) for item, users in readers.items()}, {
        'rows': rows, 'users': len(seen), 'transitions': sum(edges.values()),
        'edges': len(edges), 'sha256': digest.hexdigest(),
    }


def head_items(popularity):
    items = sorted((n for n in popularity if popularity[n] > 0),
                   key=lambda n: (-popularity[n], n))
    return set(items[:math.ceil(len(items) * .2)])


def score_edges(counts, method, shrinkage):
    if method not in METHODS or not math.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError('invalid neighbor method or shrinkage')
    outgoing, incoming = Counter(), Counter()
    for (src, dst), count in counts.items():
        if not math.isfinite(count) or count <= 0 or int(count) != count:
            raise ValueError('edge counts must be positive integers')
        outgoing[src] += count
        incoming[dst] += count
    total = sum(outgoing.values())
    scores = {}
    for (src, dst), count in counts.items():
        if method == 'count':
            scores[src, dst] = count
        else:
            ppmi = max(0., math.log(count * total / (outgoing[src] * incoming[dst])))
            scores[src, dst] = ppmi * (count / (count + shrinkage)
                                       if method == 'shrunk_ppmi' else 1)
    return scores


def rank_incoming(counts, method, shrinkage):
    scores = score_edges(counts, method, shrinkage)
    neighbors = defaultdict(list)
    for src, dst in counts:
        neighbors[dst].append(src)
    for dst, sources in neighbors.items():
        # Stable ties preserve graph edge insertion order, shared by all methods.
        sources.sort(key=lambda src: (-scores[src, dst], -counts[src, dst]))
    return neighbors


def neighbor_cache_name(cfg):
    method, shrinkage = cfg.neighbor_method, cfg.pmi_lambda
    score_edges({}, method, shrinkage)
    suffix = f'_{float(shrinkage):g}' if method == 'shrunk_ppmi' else ''
    return f'news_neighbors_{method}{suffix}.bin'


def graph_metrics(counts, popularity, method, shrinkage, k):
    if k < 1:
        raise ValueError('k must be positive')
    neighbors = rank_incoming(counts, method, shrinkage)
    baseline = rank_incoming(counts, 'count', 0)
    scores, head = score_edges(counts, method, shrinkage), head_items(popularity)
    selected, fallback, singletons, replaced = [], 0, 0, 0
    macro_pop, macro_head = [], []
    for dst, sources in neighbors.items():
        chosen = sources[:k]
        selected.extend(chosen)
        macro_pop.append(sum(math.log1p(popularity.get(n, 0)) for n in chosen) / len(chosen))
        macro_head.append(sum(n in head for n in chosen) / len(chosen))
        fallback += sum(scores[src, dst] == 0 for src in chosen) if method != 'count' else 0
        singletons += sum(counts[src, dst] == 1 for src in chosen)
        replaced += len(set(chosen) - set(baseline[dst][:k]))
    size = len(selected)
    return {
        'method': method, 'lambda': shrinkage if method == 'shrunk_ppmi' else None,
        'k': k, 'active_destination_nodes': len(neighbors), 'selected_slots': size,
        'mean_log_popularity': sum(math.log1p(popularity.get(n, 0)) for n in selected) / size if size else None,
        'head_neighbor_ratio': sum(n in head for n in selected) / size if size else None,
        'macro_mean_log_popularity': sum(macro_pop) / len(macro_pop) if macro_pop else None,
        'macro_head_neighbor_ratio': sum(macro_head) / len(macro_head) if macro_head else None,
        'fallback_ratio': fallback / size if size else None,
        'singleton_ratio': singletons / size if size else None,
        'replacement_rate': replaced / size if size else None,
        'unique_selected_neighbors': len(set(selected)), 'head_items': len(head),
    }
