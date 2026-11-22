# GLORY-PMI

**같은 GLORY 모델에서 뉴스 그래프의 이웃 선택만 바꾸는 실험 코드입니다.**
MIND-small과 Adressa-1week에 공통 실행기를 사용합니다. 원논문 성능 재현이나 PMI의 개선 효과가 이미 확인됐다는 주장은 하지 않습니다.

기반: [GLORY 공식 코드](https://github.com/tinyrolls/GLORY), commit `eff034653657486f630ff3061dcf5d7c4003a4c0` · [논문](https://arxiv.org/abs/2307.06576).
원저자 모델 파일은 수정하지 않았습니다. 데이터·GloVe·체크포인트·사용자별 예측은 Git에 올리지 않습니다.

## 브랜치와 팀 작업

| 브랜치 | 역할 |
|---|---|
| `main` | 원본 GLORY 기준점. 직접 커밋하거나 PR을 임의로 병합하지 않습니다. |
| `feat/pmi-neighbor-experiments` | 공통 전처리·학습·평가 코드와 README |
| `experiment/count` | 횟수 기반 이웃 선택 |
| `experiment/ppmi` | PPMI 기반 이웃 선택 |
| `experiment/shrunk-ppmi` | 신뢰도 수축 보정 PPMI |

실험 브랜치는 공통 코드에서 분기하고 `experiment.json`의 기본 방법만 다릅니다. 공통 수정은 공통 브랜치에 PR로 반영한 뒤 모든 실험 브랜치에 동일하게 적용합니다. 데이터셋마다 코드를 복제하지 않습니다. 명시적 `--method`는 브랜치 기본값보다 우선합니다.

```bash
git clone https://github.com/SEOWOOHYUN/glory-pmi.git
cd glory-pmi
git switch experiment/count
```

저장소는 private입니다. 팀원은 저장소 접근 권한이 있어야 합니다.

## 실행 환경

학습에는 **CUDA GPU 1개**가 필요합니다. Colab GPU 런타임 또는 Linux CUDA 환경에서 실행하세요. Mac/CPU에서는 전처리·그래프 분석·테스트를 실행할 수 있습니다. CUDA 전체 학습은 아직 이 작업 환경에서 검증하지 못했습니다.

```bash
python -m pip install -r requirements.txt
# 설치된 환경을 실험 기록에 보관
python -m pip freeze > environment-lock.txt
```

CPU 테스트 확인 환경: Python 3.9, PyTorch 2.8.0, PyG 2.6.1, NumPy 1.26.4, scikit-learn 1.6.1. CUDA용 PyTorch는 런타임에 맞게 설치하고, 모든 실험에서 같은 환경을 쓰세요. 원본의 `requirements.txt`는 버전 고정 파일이 아닙니다.

Colab에서는 아래 셸 명령 앞에 `!`를 붙이고 저장소로 `%cd` 하세요. 런타임 종료에 대비해 `--output`으로 마운트한 Drive 경로를 지정할 수 있습니다. 데이터 캐시는 동시에 쓰지 말고 실험을 순서대로 실행하세요.

## 데이터 준비

### MIND-small

[공식 MIND 페이지](https://msnews.github.io/)가 안내하는 경로에서 `MINDsmall_train.zip`, `MINDsmall_dev.zip`을 받습니다. 연결된 Hugging Face 미러는 로그인과 접근 동의가 필요할 수 있습니다. 인증 정보는 저장소에 넣지 않습니다.

```bash
python scripts/prepare_data.py mind-small \
  --train-zip data/downloads/MINDsmall_train.zip \
  --dev-zip data/downloads/MINDsmall_dev.zip
```

영어 단어 임베딩은 [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)의 `glove.840B.300d.txt`를 `data/glove/glove.840B.300d.txt`에 둡니다.
공식 dev를 체크포인트와 λ 선택에 사용합니다. 전처리 호환용 `test/`는 dev 복사본이며 **독립 테스트가 아닙니다**. 실행기는 MIND-small의 `--split test`를 거부합니다.

### Adressa-1week

[NTNU 공식 데이터](https://reclab.idi.ntnu.no/dataset/)의 이용 조건을 확인하고 one-week 아카이브를 사용합니다.

```bash
mkdir -p data/downloads
curl -fL -C - https://reclab.idi.ntnu.no/dataset/one_week.tar.gz \
  -o data/downloads/one_week.tar.gz
python scripts/prepare_data.py adressa-1week \
  --archive data/downloads/one_week.tar.gz --seed 3407 --negatives 20
```

이 저장소의 명시적 시간 분할입니다. 원논문의 전처리를 정확히 복원했다는 의미는 아닙니다.

- CET 기준 2017-01-01~04는 초기 이력, 05일은 train, 06일은 val, 07일은 test입니다.
- 기사 ID·제목·사용자·시간이 있는 이벤트를 사용하며 동일 `(사용자, 기사, 시간)` 중복을 제거합니다.
- 각 대상 클릭의 이력에는 **엄격히 이전 시각**의 클릭만 포함합니다. 같은 시각의 클릭은 서로의 이력에 포함하지 않습니다.
- 그 시점 이전에 관측된 기사 중 이력과 정답을 제외하고 최대 20개 음성 후보를 샘플링합니다. 풀에 충분한 기사가 없으면 후보가 줄어듭니다. 이력이나 음성 후보가 없으면 제외하고 개수를 기록합니다.
- 후보는 한 번 만들어 모든 방법·학습 시드에서 공유합니다. 후보 생성 시드 3407과 모델 학습 시드를 구분하세요. 학습은 이 후보에서 `npratio=4`를 사용합니다.
- Norwegian 임베딩은 원본처럼 무작위 초기화하고 엔터티를 사용하지 않습니다. MIND는 GloVe와 엔터티를 사용합니다. 두 데이터셋의 절대 점수보다 **각 데이터셋 내 Count 대비 변화량**을 비교하세요.

전처리는 기존 출력 폴더를 덮어쓰지 않습니다. `preparation.json`에 원본 해시·분할·후보 생성 조건이 저장됩니다. `train/val/test` 각각 `news.tsv`, `behaviors.tsv`, `entity_embedding.vec`를 갖는 공통 형식입니다.

## 먼저 확인할 것: 이웃이 실제로 바뀌는가?

GPU 없이 모든 방법과 λ를 한 번에 비교합니다.

```bash
python scripts/audit_graph.py \
  --behaviors data/MINDsmall/train/behaviors.tsv \
  --output runs/mind-graph.json --k 8
python scripts/audit_graph.py \
  --behaviors data/AdressaSmall/train/behaviors.tsv \
  --output runs/adressa-graph.json --k 8
```

Count, PPMI, shrinkage λ∈{1,5,20,100}의 이웃 평균 log 인기도, Head 비율, Count 대비 교체율, 1회 관측 간선 비율, 0점 fallback 비율을 기록합니다. 그래프 결과만으로 추천 정확도나 최종 추천 편향이 개선됐다고 결론 내리지 않습니다.

## 같은 조건으로 학습하기

아래 명령에서 **`--dataset`만 바꾸면** 같은 실험을 다른 데이터셋에서 실행합니다.

```bash
python scripts/run_experiment.py --dataset mind-small --method count --seed 3407 --prepare
python scripts/run_experiment.py --dataset mind-small --method ppmi --seed 3407
python scripts/run_experiment.py --dataset mind-small --method shrunk_ppmi --lambda 5 --seed 3407

python scripts/run_experiment.py --dataset adressa-1week --method count --seed 3407 --prepare
python scripts/run_experiment.py --dataset adressa-1week --method ppmi --seed 3407
python scripts/run_experiment.py --dataset adressa-1week --method shrunk_ppmi --lambda 5 --seed 3407
```

첫 실행이나 원본 데이터 변경 후에는 `--prepare`를 사용합니다. 기존 전처리의 데이터 해시가 확인되지 않거나 달라지면 재생성을 요구합니다. 실패한 실행 폴더를 덮어쓰지 않으므로 재시도는 `--output runs/새이름`을 지정하세요. `--dry-run`은 준비된 TSV에서 실행 명령만 확인합니다. 캐시와 체크포인트는 본인이 생성한 신뢰할 수 있는 파일만 사용하세요.

| 공통 조건 | 값 |
|---|---|
| 배치 / gradient accumulation | 16 / 2 |
| optimizer / 학습률 | Adam / 0.0002 |
| 최대 epoch / 조기 종료 | 5 / validation AUC 5회 연속 미개선 |
| 검증 | 2 epoch 이후 1,000 미니배치마다 + 마지막 미니배치 |
| seed | 기본 3407, 최종 비교는 3407·2024·42 권장 |
| 이웃 / hop / 그래프 | 8 / 2 / directed consecutive |
| 학습 음성 샘플 수 / dropout | 4 / 0.2 |
| 모델 선택 | validation AUC |

기울기는 원본 코드처럼 **미니배치 평균 loss를 두 번 합산**합니다. loss를 2로 나누지 않으므로 일반적인 평균 loss 배치 32와 완전히 같다고 설명하면 안 됩니다. 모든 방법에 같은 동작을 적용합니다. 마지막 남은 누적 기울기도 업데이트합니다.

같은 seed끼리 Count↔PPMI↔보정 PPMI를 비교합니다. λ는 1·5·20·100 전체를 validation에서 평가하고 선택 기준을 고정합니다. 최종 판단은 최소 3개 학습 시드의 평균과 변동을 함께 봅니다. 별도 seed로 후보를 다시 만들지 마세요.

## 저장 결과와 평가

각 `runs/<dataset>-<method>-lambda<값>-seed<값>/`에 다음을 남깁니다.

- `manifest.json`: 명령, 데이터 SHA-256, Git commit, 미커밋 변경 여부, seed.
- `config.yaml`: 해석된 전체 설정.
- `checkpoints/*_best.pth`, `best_step.txt`: 최고 validation AUC 모델과 선택 시점.
- `val-best.json`: AUC, MRR, nDCG@5/@10, Top-10 Head 비율과 후보 대비 증가량.
- `val-best.jsonl`: impression/user ID, 후보, 정답, 점수, 지표. 비교용이며 외부에 공개하지 않습니다.
- `val-latest.*`: 마지막 검증 결과. best와 혼동하지 마세요.

Adressa의 λ·모델 선택이 끝난 후 선택된 체크포인트로 test를 **한 번** 평가합니다. `--method`, `--lambda`, `--seed`는 해당 체크포인트와 일치해야 합니다.

```bash
python scripts/run_experiment.py --dataset adressa-1week \
  --method shrunk_ppmi --lambda 5 --seed 3407 --split test \
  --checkpoint runs/adressa-1week-shrunk_ppmi-lambda5-seed3407/checkpoints/GLORY_AdressaSmall_best.pth
```

두 실행의 **같은 split·seed·후보 집합**을 비교하세요. 아래는 MIND validation 예입니다.

```bash
python scripts/compare_runs.py \
  --baseline runs/mind-small-count-lambda5-seed3407/val-best.jsonl \
  --treatment runs/mind-small-shrunk_ppmi-lambda5-seed3407/val-best.jsonl \
  --output runs/mind-paired-seed3407.json
```

차이는 treatment−Count입니다. 사용자 단위 paired bootstrap 2,000회로 impression 평균 차이의 95% 구간을 계산합니다. 반복 사용자의 상관을 보존하지만 **학습 시드 변동이나 validation 선택 편향까지 제거하지는 않습니다**. 후보/정답/사용자가 다르면 비교를 거부합니다.

## 구현된 비교의 정확한 의미

전이는 원본 GLORY와 동일하게 **학습 TSV에서 사용자별 첫 행의 이력**만 사용합니다. 모든 행의 중복 이력을 합산하지 않습니다. `i→j` 간선에서 `cᵢⱼ`, 출발합 `oᵢ`, 도착합 `dⱼ`, 총전이 `N`을 계산합니다.

```text
Count       = c_ij
PPMI        = max(0, log(c_ij * N / (o_i * d_j)))
Shrunk PPMI = c_ij / (c_ij + lambda) * PPMI
```

원본 모델은 j의 **들어오는 이웃 i**를 선택합니다. 이를 유지하며 신경망에 PMI를 곱하지 않고 이웃 순서만 바꿉니다. 원래 정수 간선 횟수와 엔터티 그래프를 보존합니다. 점수 동률은 횟수, 다시 동률이면 원래 간선 순서로 결정합니다. 양수 점수가 모자라면 0점 간선을 횟수순으로 채우며, 없는 간선을 만들거나 padding을 이웃 통계에 넣지 않습니다.

인기도는 모든 학습 이력과 학습 positive 후보에 등장한 **서로 다른 사용자 수**입니다. 양의 인기도를 가진 기사 중 상위 ceil(20%)가 Head이며 동률은 기사 ID로 고정합니다. 학습에서 보지 못한 기사는 0/비Head입니다. 그래프는 선택된 슬롯 기준 평균과 노드별 평균을 모두 저장합니다.

평가 지표는 impression 평균입니다. Top-10은 후보가 부족하면 `min(10, 후보수)`를 씁니다. 후보≤10이면 Head 증가량은 항상 0이므로 후보>10 부분집합 지표도 따로 보고합니다. 단일 정답 클래스만 있는 impression의 AUC는 제외하고 유효 개수를 남깁니다. MRR은 MIND 방식으로 모든 positive의 reciprocal rank 평균이며, 점수 동률은 입력 후보 순서를 유지합니다. 빈 이력은 모든 방법에서 일관되게 제외합니다.

**효과 판단:** 이웃 Head 집중 감소 → 추천 Head 집중 감소 → 정확도 변화 순서로 확인합니다. 정확도의 신뢰구간이 0을 포함한다는 사실만으로 “정확도 유지”를 입증할 수 없습니다. 유지 주장을 하려면 실험 전에 허용 하락폭을 정하고 비열등성을 검토하세요. MIND dev는 탐색적 검증 결과로 표시합니다. Adressa 음성 후보는 실제 노출 로그가 아니므로 MIND와 Head 비율을 직접 비교하지 않습니다. 측정 대상은 관측 클릭 인기도 집중이며 인과적 편향 제거가 아닙니다.

## 개발 검증

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

테스트는 수식·fallback·캐시 분리·빈 이력·시간 분할·공통 실행 설정·paired 비교와 두 데이터셋 모델의 합성 CPU 순전파/역전파/평가를 확인합니다. 실제 CUDA 학습 성공이나 PMI 효과를 보증하는 테스트는 아닙니다. NRMS/LSTUR 추가 학습은 현재 구현 범위에 포함하지 않았습니다.
