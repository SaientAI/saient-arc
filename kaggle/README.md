# ARC-AGI-2 notebook preparation

This repository notebook embeds version 0.2 of the corrected public symbolic solver. It is a preparation artifact, with no ARC-AGI-2 hidden-score claim. Updating it here does not replace any existing Kaggle notebook version or submission.

The notebook contains only the four public Python modules. It locates `arc-agi_test_challenges.json` in the attached Kaggle inputs, runs the three representation views, and writes `/kaggle/working/submission.json` with two attempts per test input.

The rebuilt notebook was smoke-tested in a fresh network-isolated process on three synthetic tasks with four TEST inputs. All outputs had two valid attempts; this checks execution and schema, not ARC-AGI-2 accuracy or on-platform acceptance.

Run `python3 tools/build_kaggle_bundle.py --check` from the repository root to verify all embedded source bytes. Run without `--check` to rebuild after source changes. The builder preserves all other notebook content.

For use on Kaggle, attach the competition data and inspect the completed run and generated submission file before submitting. Refer to the [competition page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) for its current participation and submission requirements.

Public source: https://github.com/SaientAI/saient-arc
