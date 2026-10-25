# Saient ARC Gremlin

Open-source symbolic ARC reasoning research from **SaientAI**.

Saient ARC Gremlin explores ARC-AGI-1 puzzles through bounded hypothesis selection, geometric/object transformations, and optional spatial-program synthesis. Offline Python 3.12 standard-library only; no model weights or external APIs.

## Tested public results (ARC-AGI-1, not ARC-AGI-2)

| Method | Public evaluation tasks | Exact output grids |
|---|---:|---:|
| Original ARC Gremlin | 2/400 | 2/419 (0.4773%) |
| Multi-representation selector | **3/400** | **3/419 (0.7160%)** |

The multi-representation system applies the same solver to its original grid, a consistently rotated grid, and a canonically recoloured grid. It scores candidates using only labelled *training examples*, selects by exact/pixel training agreement, and maps predictions back. The extra public evaluation solve was task e41c6fd3 via palette normalisation. Scores are **self-reported on the public ARC-AGI-1 evaluation set**, not privately verified. No AGI claims.

A separate opt-in prototype (arc_gremlin/spatial_programs.py) adds connected-component scene graphs, spatial relationships and a safe two-step transformation DSL. arc_gremlin/extended_solver.py adds budgeted test-time search. This is **off by default** and not included in the 0.7160% score. On 40 public ARC-AGI-1 *training* tasks, combined extended solving did **not** improve upon the 3/40 baseline and used more CPU time. The six synthetic designed tasks are for development only, not independent ARC evidence.

## Reproduce

1. Obtain ARC-AGI-1 v1 public data from https://github.com/fchollet/ARC-AGI .
2. Set SAIENT_ARC_DATA to the folder with 400 public evaluation task JSON files (usually data/evaluation).
3. Run: python3 tools/benchmark.py . It writes results/eval_summary.json and results/eval_results.json. Optional SAIENT_ARC_LIMIT=2 is a smoke test only.
4. Run: python3 -m unittest discover -s tests -v to check the spatial research components.

Ground-truth test outputs are used only for post-prediction scoring, never as inputs to the solver or selector. Per-output score is 3/419 (0.71599%); task-level score is 3/400 (0.75%). Full three-view evaluation took about 135–148 seconds locally (hardware-dependent).

## Optional spatial program search

Example: python3 -m arc_gremlin.extended_solver task.json --multi-view --program-search --budget-ms 1000 --max-candidates 512

The Python-level deadline is not a hard operating-system deadline. The synthesis DSL uses whitelisted transformations; no arbitrary generated Python execution.

## Licence

MIT License. Copyright 2026 SaientAI contributors. See LICENSE.

This is the standalone public ARC research implementation, not a release of unrelated software.

## Leaderboard status

Submitted for community review at https://github.com/arcprize/ARC-AGI-Community-Leaderboard . Inclusion is discretionary. This code does not claim scores for the ARC-AGI-2/3 Kaggle competitions.
