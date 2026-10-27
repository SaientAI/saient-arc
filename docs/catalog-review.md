# Public catalog correction — 27 October 2026

This revision responds to the general-purpose concern in [community leaderboard PR #63](https://github.com/arcprize/ARC-AGI-Community-Leaderboard/pull/63#issuecomment-6451141986). Matching a predefined answer recipe to training examples does not establish that its construction was inferred from those examples.

The audit covered the complete symbolic solver, all hypothesis registries and tier entry points, optional spatial/extended modules, the benchmark runner, and the notebook's embedded source. Removal decisions were based on source semantics before evaluating the corrected catalog. Lower scores were explicitly acceptable; no removed rule was reinstated based on evaluation results.

## Removed procedures

The following procedures, their inference wrappers and unused helpers were removed entirely, including registrations in both the base and relational catalogs. There is no switch that enables the old recipes in this release.

| Family | Reason for removal |
| --- | --- |
| Diagonal and bottom bar | Fixed output colors 2/4 and a prescribed drawing recipe. |
| Three-panel triptych | Fixed 3×11 canvas, 3×3 panels, rotations and separator color 5. |
| Triad majority pattern, both variants | Fixed 3×3 mapping from row-color relationships to a row or diagonal glyph. Learning just the marker color did not infer that glyph construction. |
| Row-position color rows | Learned a compact position-to-color mapping, but restricted it to a fixed 3×3 recipe. Removed conservatively; this was not whole-grid memorization. |
| Two-row stripe interleaving | Prescribed two-row alternating layout, removed conservatively. |
| Marker frame fill | Bespoke frame construction with an unexplained two-row offset. |
| Non-anchor recoloring to five | Fixed destination color 5. |
| Uniform-row-anchor preservation and recoloring | Fixed destination color 5 and prescribed anchor recipe. |
| Sliding diagonal from a five-cell row | Fixed 1×5 input, scale 5 and diagonal stamping layout. |
| Separator-block labels | Fixed separator color 5 and region colors 1/2/3. |
| Marker sequence rewrite | Stored whole training inputs and their corresponding output grids; returned identity for unseen inputs. |

The source deletion covers 24 functions, including wrappers and helpers. The B+ lane also no longer privileges five exact conditional-program names. Ordinary conditionals are still generated from the current task.

## Corrected parameter inference

Biaxial marker expansion now requires an observed crossing color that is consistent across the task's training examples, and its resulting transformation must exactly reproduce every training output. There is no default color 2 and no admission based on partial agreement. If the parameter is unidentifiable or the examples conflict, this candidate is absent.

Uniform-row marking already inferred its output marker from the examples. Its unrelated 3×3 registration gate has been removed. Tests cover transfer to different dimensions and input palettes, differing learned output colors, and rejection of conflicting examples. These are synthetic regression tests, not benchmark scores.

## Retained search and assumptions

The solver remains a bounded, handwritten symbolic hypothesis search. It retains generic rotations, reflections, translations, bounding-box crops, scale/tiling, repetition, connected-component operations, spatial relationships, parameter inference from the current task, and bounded compositions/conditionals. Examples include pointwise color maps, symmetry-to-label colors, repetition lengths and uniform-row marker colors learned from TRAIN. There is no claim that every retained operation or architectural choice is learned automatically.

Background color 0, four-neighbor connectivity, bounded scale/translation ranges, geometric choices, object-selection heuristics, fit thresholds and search budgets remain explicit inductive biases. Directional operations need not be rotation-equivariant. Caches retain computed features and hypotheses for a task; they are not a shipped database of solved tasks. The optional spatial DSL enumerates a bounded set of primitive programs and is excluded from the reported three-view benchmark.

The two functions named by the reviewer can be traced in the local history to an earlier bulk source snapshot, without their individual development steps. That history does not establish which particular puzzles prompted them. We do not claim a known clean origin for the old catalog.

## Measurement and packaging

The benchmark now finishes all three predictions and selects a view using only TRAIN fit before reading TEST outputs for scoring. Its original/rotated/palette views, six-decimal TRAIN pixel ranking and original-view tie preference are retained. Synthetic tests check that TEST labels cannot affect the selected predictions.

`tools/build_kaggle_bundle.py` rebuilds the notebook's four embedded Python modules deterministically; `--check` verifies that each embedded byte matches the current source. Updating this repository does not replace or submit an existing Kaggle run.

An earlier narrow ablation disabled only the two named registrations at revision 69e7a79236c1fe181ac365a6a02a08387c503173. All 400 tasks' predictions were unchanged: 3/400 complete tasks and 3/419 output grids in both runs. That narrow result is distinct from the full catalog correction reported in the README and release evidence.

The public evaluation has already been seen during development. Its score is descriptive reproduction evidence, not a new unseen-data estimate. This audit and its tests do not certify generalization or guarantee leaderboard acceptance. The maintainers decide whether the corrected submission satisfies their criteria.
