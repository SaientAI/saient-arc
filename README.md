# Saient ARC Gremlin

Open-source symbolic ARC reasoning research from **SaientAI**. Version **0.2** corrects the hypothesis catalog following community review.

The solver uses bounded geometric and object transformations, task-derived parameters, and training-example fit. A deterministic selector compares original, rotated and palette-normalized representations. Python 3.12 standard library only; no model weights or external inference APIs.

## Catalog correction

Version 0.2 removes fixed puzzle recipes, whole-grid training-example lookup and an exact-program priority list. It also rejects unsupported crossing-color inference and removes an arbitrary 3×3 restriction from learned uniform-row marking. The notebook's embedded source has been rebuilt from the same corrected modules.

See the [catalog audit](docs/catalog-review.md) for the removed families, retained assumptions, provenance limits and scope. Handwritten primitives, bounded search and heuristics remain. This correction is not a claim of proven generalization or guaranteed leaderboard acceptance.

## Reproduced public results

The complete ARC-AGI-1 public evaluation was rerun on 27 October 2026 after freezing the correction:

| Version 0.2 method | Complete tasks | Exact output grids |
| --- | ---: | ---: |
| Original-view solver | 2/400 | 2/419 (0.4773%) |
| Three-view selector | **3/400** | **3/419 (0.71599%)** |

The corrected selector has the same task/grid totals as version 0.1, with no gained or lost tasks. One unsolved task's prediction changed. All 400 tasks were evaluated, with 419 test grids and zero view errors; the full run took 131.5 seconds locally. Runtime depends on hardware and load.

[Machine-readable evaluation evidence](docs/evaluation-v0.2.json) includes source and dataset hashes, per-task outcomes, score denominators and comparison with the prior revision. Test outputs are used only after predictions and TRAIN-based view selection are complete. These are self-reported results on public data already seen during development, not a new unseen-data estimate or a privately verified score. No ARC-AGI-2/3 score is claimed.

## Reproduce

1. Obtain [ARC-AGI-1 public data](https://github.com/fchollet/ARC-AGI). The evidence file records the dataset commit and individual file hashes.
2. Set `SAIENT_ARC_DATA` to its `data/evaluation` directory containing 400 JSON files.
3. Run `python3 tools/benchmark.py`. It writes `results/eval_summary.json` and `results/eval_results.json`; `SAIENT_ARC_OUT` can choose a different destination. `SAIENT_ARC_LIMIT` is only for smoke tests and must be unset for the full evaluation.
4. Run `python3 -m unittest discover -s tests -v` (23 tests in this revision).
5. Run `python3 tools/build_kaggle_bundle.py --check` to verify that the notebook embeds the current source. Rebuild it with the same command without `--check` after source changes.

## Optional spatial program search

`arc_gremlin/spatial_programs.py` contains connected-component scene graphs, spatial relations and a bounded two-step DSL. `arc_gremlin/extended_solver.py` adds budgeted test-time search. Both are opt-in and excluded from the reported 0.71599% result. Synthetic tests exercise their behavior; they are not independent ARC accuracy evidence.

Example: `python3 -m arc_gremlin.extended_solver task.json --multi-view --program-search --budget-ms 1000 --max-candidates 512`.

The Python-level deadline is not a hard operating-system deadline. The synthesis DSL uses whitelisted transformations and does not execute arbitrary generated Python.

## Licence and review status

MIT License. Copyright 2026 SaientAI contributors. See [LICENSE](LICENSE).

This is the standalone public ARC implementation. [Community leaderboard PR #63](https://github.com/arcprize/ARC-AGI-Community-Leaderboard/pull/63) remains subject to maintainer review of general-purpose behavior, openness and novelty. Updating this repository does not replace or submit an existing Kaggle run.
