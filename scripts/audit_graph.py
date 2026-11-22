"""CPU-only real graph audit; accepts GLORY-format training behaviors."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dataload.neighbor_scoring import read_training, graph_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--behaviors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=8)
    args = parser.parse_args()
    edges, popularity, info = read_training(args.behaviors)
    results = [graph_metrics(edges, popularity, method, shrinkage, args.k)
               for method, shrinkage in [('count', 0), ('ppmi', 0)] +
               [('shrunk_ppmi', value) for value in (1, 5, 20, 100)]]
    payload = {'input': str(args.behaviors), 'data': info, 'results': results,
               'scope': 'graph-only; not recommendation accuracy; first history per user; incoming neighbors'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
