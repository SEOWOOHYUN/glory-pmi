"""Paired comparison of prediction exports (treatment minus baseline)."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.experiment_metrics import paired_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2000)
    args = parser.parse_args()
    runs = []
    for path in (args.baseline, args.treatment):
        with path.open() as stream:
            runs.append([json.loads(line) for line in stream])
    result = [paired_bootstrap(*runs, key, args.repeats) for key in
              ('auc', 'mrr', 'ndcg5', 'ndcg10', 'head_ratio10', 'head_amplification10')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
