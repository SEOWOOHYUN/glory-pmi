"""Prepare local MIND archives or raw Adressa-1week for the shared GLORY runner."""
import argparse
import bisect
import hashlib
import json
import random
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
from pathlib import Path
from itertools import groupby


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_mind(train_zip, dev_zip, output):
    # Extract only known plain data members, never arbitrary archive paths/pickles.
    archives = {'train': zipfile.ZipFile(train_zip), 'val': zipfile.ZipFile(dev_zip)}
    with ExitStack() as stack:
        for archive in archives.values():
            stack.enter_context(archive)
            for name in ('news.tsv', 'behaviors.tsv', 'entity_embedding.vec'):
                archive.getinfo(name)
        output.mkdir(parents=True, exist_ok=False)
        for split, archive in archives.items():
            (output / split).mkdir()
            for name in ('news.tsv', 'behaviors.tsv', 'entity_embedding.vec'):
                with archive.open(name) as source, (output / split / name).open('wb') as dest:
                    shutil.copyfileobj(source, dest)
        # Upstream preprocessing expects test metadata; this is NOT a held-out test.
        shutil.copytree(output / 'val', output / 'test')
    metadata = {'dataset': 'MIND-small', 'test_is_validation_copy': True,
                'train_zip_sha256': sha256(train_zip), 'dev_zip_sha256': sha256(dev_zip)}
    (output / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n')


def read_adressa(archive_path):
    events, titles, title_times = [], {}, {}
    stats = Counter()
    days = set()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or not Path(member.name).name.startswith('201701'):
                continue
            days.add(Path(member.name).name)
            print(f'Reading {member.name}', flush=True)
            stream = archive.extractfile(member)
            for raw in stream:
                stats['raw_events'] += 1
                row = json.loads(raw)
                # URL-only records include non-article page views; require article ID/title.
                if not all(row.get(key) for key in ('userId', 'id', 'title', 'time')):
                    stats['missing_article_fields'] += 1
                    continue
                title = row['title']
                if isinstance(title, list):
                    title = ' '.join(str(part) for part in title)
                if not isinstance(title, str) or not title.strip():
                    stats['empty_title'] += 1
                    continue
                timestamp = int(row['time'])
                item, user = str(row['id']), str(row['userId'])
                events.append((user, item, timestamp))
                if item not in title_times or timestamp < title_times[item]:
                    title_times[item] = timestamp
                    titles[item] = ' '.join(title.split())
    if days != {f'201701{day:02d}' for day in range(1, 8)}:
        raise ValueError(f'Expected all seven Adressa day files, found {sorted(days)}')
    return events, titles, dict(stats)


def write_adressa(events, titles, output, seed=3407, negatives=20):
    if negatives < 1:
        raise ValueError('negatives must be positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing data: {output}')
    # Source files use Norwegian local dates (CET, UTC+1 in January 2017).
    tz = timezone(timedelta(hours=1))
    by_user, first_seen = defaultdict(list), {}
    duplicate_count = len(events) - len(set(events))
    for user, item, timestamp in sorted(set(events), key=lambda row: (row[2], row[0], row[1])):
        day = datetime.fromtimestamp(timestamp, tz).strftime('%Y%m%d')
        if not '20170101' <= day <= '20170107':
            continue
        by_user[user].append((timestamp, item))
        first_seen[item] = min(first_seen.get(item, timestamp), timestamp)
    ids = {item: f'N{i + 1}' for i, item in enumerate(sorted(first_seen))}
    available = sorted(first_seen, key=lambda item: (first_seen[item], item))
    times = [first_seen[item] for item in available]
    rng = random.Random(seed)
    counts, skipped = Counter(), Counter()
    output.mkdir(parents=True)
    with ExitStack() as stack:
        files = {}
        for split, last_day in [('train', '20170105'), ('val', '20170106'), ('test', '20170107')]:
            folder = output / split
            folder.mkdir()
            files[split] = stack.enter_context((folder / 'behaviors.tsv').open('w'))
            with (folder / 'news.tsv').open('w') as news:
                for item in sorted(ids):
                    if datetime.fromtimestamp(first_seen[item], tz).strftime('%Y%m%d') <= last_day:
                        news.write('\t'.join([ids[item], 'news', 'news', titles[item], '', '', '[]', '[]']) + '\n')
            (folder / 'entity_embedding.vec').write_text('')
        for user_index, user in enumerate(sorted(by_user), 1):
            history = []
            for timestamp, simultaneous in groupby(by_user[user], key=lambda event: event[0]):
                items = [event[1] for event in simultaneous]
                for item in items:
                    day = datetime.fromtimestamp(timestamp, tz).day
                    if day >= 5:
                        split = {5: 'train', 6: 'val', 7: 'test'}[day]
                        excluded = set(history) | {item}
                        end = bisect.bisect_left(times, timestamp)
                        # Uniform sampling from prior available articles, without future-click exclusion.
                        chosen = set()
                        limit = min(negatives, end)
                        for _ in range(max(100, limit * 20)):
                            if len(chosen) == limit or not end:
                                break
                            candidate = available[rng.randrange(end)]
                            if candidate not in excluded:
                                chosen.add(candidate)
                        if len(chosen) < limit:
                            pool = [n for n in available[:end] if n not in excluded and n not in chosen]
                            chosen.update(rng.sample(pool, min(limit - len(chosen), len(pool))))
                        if chosen and history:
                            candidates = [f'{ids[item]}-1'] + [f'{ids[n]}-0' for n in sorted(chosen)]
                            rng.shuffle(candidates)
                            counts[split] += 1
                            row = [f'{split}-{counts[split]}', f'U{user_index}', str(timestamp),
                                   ' '.join(ids[n] for n in history), ' '.join(candidates)]
                            files[split].write('\t'.join(row) + '\n')
                        else:
                            skipped[split] += 1
                history.extend(items)
    metadata = {'dataset': 'Adressa-1week', 'protocol': 'project-cet-days1-4-history-day5-train-day6-val-day7-test',
                'candidate_seed': seed, 'sampled_negatives': negatives,
                'test_is_validation_copy': False, 'impressions': dict(counts),
                'skipped_empty_history_or_pool': dict(skipped), 'duplicate_events_removed': duplicate_count,
                'articles': len(ids), 'users': len(by_user)}
    (output / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='dataset', required=True)
    mind = sub.add_parser('mind-small')
    mind.add_argument('--train-zip', type=Path, required=True)
    mind.add_argument('--dev-zip', type=Path, required=True)
    mind.add_argument('--output', type=Path, default=Path('data/MINDsmall'))
    adressa = sub.add_parser('adressa-1week')
    adressa.add_argument('--archive', type=Path, required=True)
    adressa.add_argument('--output', type=Path, default=Path('data/AdressaSmall'))
    adressa.add_argument('--seed', type=int, default=3407)
    adressa.add_argument('--negatives', type=int, default=20)
    args = parser.parse_args()
    if args.dataset == 'mind-small':
        prepare_mind(args.train_zip, args.dev_zip, args.output)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        events, titles, stats = read_adressa(args.archive)
        info = write_adressa(events, titles, args.output, args.seed, args.negatives)
        info.update(source_sha256=sha256(args.archive), raw_filter_counts=stats)
        (args.output / 'preparation.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'Prepared {args.output}; see preparation.json')


if __name__ == '__main__':
    main()
