# GLORY PMI experiments

User objective: compare count, PPMI, and c/(c+lambda)*PPMI on MIND-small
and Adressa-1week without changing the neural architecture. Team members
must be able to run the experiments from a GitHub PR.

Use upstream eff034653657486f630ff3061dcf5d7c4003a4c0. Preserve its first
history per user graph and incoming-neighbor orientation. Preserve integer
edge counts and the entity graph; change only news neighbor order. Use
deterministic ties (score, count, original edge order). Positive scores
precede zero scores; zero scores use count fallback. Do not invent edges.
Isolate neighbor caches by method/lambda; rebuild from the loaded graph.

Population: all items clicked in training histories or positive training
impressions, unique users per item. Head = ceil(20% of these items), ties
broken by raw news ID. Unseen items have popularity zero and are not Head.
Graph measures use selected incoming slots, excluding padding and empty
nodes. Save macro node averages as well as slot averages, replacement and
fallback rates. Ranking measures average per impression and separately
report candidates >10. Raw per-impression predictions enable paired tests.

Alternatives: launch all training immediately or first audit graphs. Choose
graph audit first (agent judgment): it is inexpensive and detects changes
in the actual experimental variable. Follow with separately trained models,
same seeds and candidates. lambda in {1,5,20,100}, tune on validation AUC;
report all choices, no test-based tuning. At least 3 seeds for final claims.
Do not claim noninferiority without a prespecified accuracy margin.

Use batch_size=16 and accumulation_steps=2 as requested, preserving upstream
summed gradients and documenting that this is not mean-loss batch-32 equivalence.
Fix empty-history evaluation duplication in all arms. Bound training batches.
Single-GPU reference path; reject multi-GPU for this experimental runner.

MIND official dev is validation, not independent test (upstream copies it).
Adressa raw preparation is a declared project protocol, not an exact
reconstruction of unpublished upstream splits: CET days 1-4 initial history, day 5 train,
day 6 validation, day 7 test, only earlier clicks in each history; sampled
negatives from articles observed before the impression. Persist candidates.
Do not compare sampled Adressa Head rates directly with MIND exposure lists.

Scope: scoring, graph CLI, preparation, loaders, training/evaluation exports,
tests, Colab commands, reproducible reports. NRMS/LSTUR are contextual
baselines for later; the direct PMI ablation is the first deliverable.
No claims of recommendation gains from graph-only results or synthetic data.
