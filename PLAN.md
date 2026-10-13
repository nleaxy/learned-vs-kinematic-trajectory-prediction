# Pre-experiment plan

Fixed on: 2026-10-14, before any results were obtained.

## Research questions
- RQ1: How does RNN prediction accuracy change as history frequency drops from 10 to 1 Hz?
- RQ2: At which frequency does the RNN lose its advantage over kinematic baselines, on straight and turning trajectories?
- RQ3: Can a model trained at 10 Hz be used on sparse inputs without retraining?

## Hypotheses (5% = negligibility threshold)
- H1: 10→5 Hz increases RNN ADE by ≤5%; at 1 Hz by >20%. 2 Hz is exploratory.
- H2: At 2 and 1 Hz, the RNN advantage over the best kinematic model (CV/CA) is <5% on straight (Δψ < 15°) and >5% on turning (Δψ > 45°) trajectories.
- H3: At 2 and 1 Hz, the 10 Hz model is worse than the model trained at the matching frequency by >5%.

## Setup
- Data: Argoverse 2 Motion Forecasting, focal agent only, no maps/neighbours. History 5 s, horizon 6 s. Train 50,000 / val 5,000 scenarios from official train; test = official validation. Only scenarios with all 110 steps observed.
- Factor: history frequency 10 / 5 / 2 / 1 Hz (main: fixed 5 s window; additional: fixed 5 points).
- Models: CV, CA, RNN per frequency; 10 Hz RNN on downsampled inputs (linear interpolation; ablation: last-value repeat, raw sparse input).
- Hyperparameters: grid hidden {64,128} × lr {1e-3, 3e-4} per frequency, selected on validation with one seed.
- Metrics: ADE (primary), FDE, MR@2m. Subgroups: agent type; straight / turning; intermediate (15–45°) reported separately. Groups <300 scenarios reported descriptively.

## Statistics
- 5 seeds (10 if seed spread >2%).
- Paired bootstrap over scenarios; 8 main comparisons, Bonferroni correction.
- Decision by position of the confidence interval of the relative difference against the threshold.