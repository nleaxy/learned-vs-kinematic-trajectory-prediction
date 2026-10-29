# Pre-experiment plan

Version 4, fixed on: 2026-10-29, after a timing run and before the main runs.
Version 3 was fixed on 2026-10-27 (commit `4984b3b`, tag `protocol-v3`) after data processing and before any model was trained; version 2 on 2026-10-26 (commit `82ce880`, tag `protocol-v2`) and version 1 on 2026-10-14 (commit `59e0891`, tag `protocol-v1`), both before any data were processed. Changes are listed at the end. After the first main run the plan is not edited.

## Research questions
- RQ1: How does RNN prediction accuracy change as history frequency drops from 10 to 1 Hz?
- RQ2: At which frequency does the RNN lose its advantage over kinematic baselines, on straight and turning trajectories?
- RQ3: Can a model trained at 10 Hz be used on sparse inputs without retraining?

## Hypotheses (5% = negligibility threshold)
- H1: 10→5 Hz increases RNN ADE by ≤5%; at 1 Hz by >20%. 2 Hz is exploratory.
- H2: At 2 and 1 Hz, the RNN advantage over the best kinematic model (CV/CA) is <5% on straight (Δψ < 15°) and >5% on turning (Δψ > 45°) trajectories.
- H3: At 2 and 1 Hz, the 10 Hz model is worse than the model trained at the matching frequency by >5%.

## Data
- Argoverse 2 Motion Forecasting, focal agent only, no maps or neighbours. History 5 s (50 points), horizon 6 s (60 points), 10 Hz.
- Test = official test split (24,984 scenarios): 5 s histories from the scenario files, 60 future steps from the single-agent test annotations released by the dataset authors (`av2_mf_focal_test_annotations.parquet`, downloaded on 2026-10-26). The official validation split is not used. The test choice is final.
- Train and val come from `configs/train_subset.txt` (60,000 ids from official train, seed 42). After the 110-step filter, the list is shuffled with seed 42: first 5,000 = val, next 50,000 = train.
- `configs/train_ids.txt`, `configs/val_ids.txt`, `configs/test_ids.txt` and `configs/norm_stats.json` are committed before any training.

## Frequency conditions
- Downsampling by k = 1, 2, 5, 10 (10, 5, 2, 1 Hz): every k-th point counting back from the last observed point (t = 0), no low-pass filter. Retained points at t = -m·k·0.1 s.
- Main mode: all retained points of the 5 s window. 50 / 25 / 10 / 5 points, span 4.9 / 4.8 / 4.5 / 4.0 s.
- Additional mode: last 5 retained points. Span 0.4 / 0.8 / 2.0 / 4.0 s.

## Models
- CV and CA from finite differences of the last 2 or 3 retained points.
- RNN per frequency and mode: 1-layer LSTM, linear decoder on the last hidden state, all 60 future points at once.
- Input frame: origin at the point at t = 0, rotated by the dataset heading of this point. The point is kept in every condition; the heading is used only for the rotation and is not an input feature. Input features per retained point: position (x, y) and the step from the previous retained point (dx, dy), zero for the first point. For the 10 Hz model on sparse inputs the steps are computed after the grid is restored; with the naive input they span k·0.1 s.
- 10 Hz RNN on sparse inputs: linear interpolation back to the 10 Hz grid (main); zero-order hold and naive sparse input (mini-ablation). Grid steps before the first retained point copy it.
- No model gets the time step as input. Every batch holds sequences of equal length, no padding. The naive input changes the grid and the time step at once and is interpreted that way.

## Training
- All settings in `configs/train.yaml`: Adam, batch 256, loss = ADE, max 50 epochs, patience 5 with no early stop in the first 10 epochs, final weights from the epoch with the lowest val ADE.
- Inputs standardized per feature with mean and std of the train histories of the same frequency and mode; the 10 Hz model on sparse inputs keeps the 10 Hz values. Targets in meters.
- Grid hidden {64, 128} × lr {1e-3, 3e-4} per frequency, selected on val with seed 0 (not reused).
- Main runs: seeds 1–5. If the sample std of val ADE over seeds 1–5 exceeds 2% of the mean at any frequency of the main mode, all four main-mode frequencies get seeds 6–10. Decided on val only, before any test evaluation. Additional mode at 10, 5, 2 Hz: main-mode hyperparameters, seeds 1–3. At 1 Hz both modes hold the same five points, so the additional mode reuses the main-mode models and results; no separate training.
- Timing run before the main runs: main mode at 10 Hz, hidden 128, lr 1e-3, seed 0, 2,000 train scenarios, full val, at most 50 epochs. It checks that training works and measures time per epoch; its numbers are not reported. If the projected time of all runs does not fit before 2026-11-03, reductions are made in this order and recorded as version 4 before the main runs: (1) drop the additional mode; (2) use the hyperparameters selected at 10 Hz for all frequencies. Nothing else is changed on this basis.

## Metrics and groups
- ADE (primary), FDE, MR@2m.
- Road user types: vehicle (incl. bus), pedestrian, two-wheeler.
- Motion types from the 10 Hz ground truth: stationary (future path < 1 m); undefined (no future point at least 2 m from the final position); otherwise Δψ is the angle between the dataset heading at t = 0 and the direction from the last future point at least 2 m from the final position to the final position: straight (Δψ < 15°), turn (Δψ > 45°), intermediate (15–45°). Only straight and turn agents are used in H2. Diagnostic grouping only: it uses the future and is unknown at prediction time.
- Quality control: 100 random test trajectories per motion type plotted (`figures/qc`), doubtful panels re-plotted in meters (`src/qc_zoom.py`), result in `results/qc_check.csv`.

## Statistics
- Per-scenario errors are saved for every run (seed) separately.
- Two-way bootstrap (Owen 2007, pigeonhole bootstrap), 10,000 resamples: one scenario draw per resample, shared by all models; seeds drawn separately for each independently trained RNN; the 10 Hz model on sparse inputs reuses the seed draw of the 10 Hz model. Δ, A, E computed from resample means. Percentile intervals. Group intervals resample scenarios within the group.
- 8 primary comparisons on all road user types together: H1 at 5 and 1 Hz, H2 straight and turn at 2 and 1 Hz, H3 at 2 and 1 Hz. Bonferroni level 1 - 0.05/8.
- Each comparison is a part of its hypothesis: supported (interval fully on the stated side), contradicted (fully on the other side), inconclusive (crosses the threshold). A hypothesis is accepted only if all its parts are supported, otherwise partly supported (at least one part) or not supported.
- Groups below 300 test scenarios are descriptive. If the straight or the turn group is below 300, its H2 parts are descriptive and H2 cannot be accepted as a whole; Bonferroni still assumes 8 comparisons.
- Road user types are secondary (95% intervals); statements about a hypothesis only for types with at least 300 test scenarios.
- Minimum sufficient frequency is a separate applied criterion, not one of the 8 comparisons: the lowest tested f with the upper 95% bound of Δ below 5% at f and at every tested frequency between f and 10 Hz. If none below 10 Hz qualifies, the result is that no tested reduction is sufficient.

## Changes in version 2 (supervisor review and our own check, before any data were processed)
1. Time axis: retained times and spans written out for every condition.
2. 10 Hz model on sparse inputs: padding, batching and the missing time-step input described.
3. Bootstrap: version 1 averaged errors over seeds and resampled scenarios only; version 2 resamples scenarios and seeds.
4. Decision rules for multi-part hypotheses and small groups written out.
5. Motion types declared a diagnostic grouping.
6. Training settings fixed in `configs/train.yaml`; seed rule and seed numbers specified.
7. Minimum sufficient frequency declared a separate criterion; condition on all higher frequencies; case with no sufficient frequency.
8. Test split: version 1 used the official validation split because the test futures were assumed to be withheld. The dataset authors have released the test annotations, so the test set is now the official test split. Split rule for the 60,000-scenario subset written out.
9. Use of the heading at t = 0 written out.
10. At 1 Hz the additional mode reuses the main-mode models (supervisor review, 20 Oct).

## Changes in version 3 (after data processing, before any model was trained)
1. Motion types: the direction at t = 0 is the dataset heading (version 2: displacement over the last 0.5 s of history, not defined for slow agents). The final direction uses the last 2 m of the future path (version 2: displacement over the last 0.5 s of the future). The quality check on 2026-10-27 showed that a short final stretch assigns stopped vehicles to turns when their recorded position drifts backward. A new group, undefined, holds agents without a 2 m final stretch and is not used in H2.
2. Timing run and the order of possible reductions written out.

## Changes in version 4 (after the timing run, before the main runs)
1. Inputs: the timing run (`results/runs/timing_f10.json`, 2,000 train scenarios) showed that the 10 Hz model with positions as the only input did not learn to use velocity: val ADE stayed at 12.3 m, against 4.3 m for CV. At 10 Hz the step between neighbouring points is about 0.06 in standardized units, too small for the model to pick up. The step from the previous retained point is added as an input, and inputs are standardized per frequency and mode, because steps at 1 Hz are ten times longer than at 10 Hz. `configs/norm_stats.json` is no longer used. A check run with 20,000 train scenarios (`results/runs/check_inputs_f10.json`) reached val ADE 3.2 m. Version 3 allowed only reductions after the timing run; this change goes beyond that and is made because training did not work. No main run had been made and the test set was not used.
2. Early stopping starts after epoch 10, since both runs spent their first epochs on a plateau.
3. The timing run showed that all runs fit before 2026-11-03, so no reductions are made: the additional mode and the per-frequency grid stay.