# E02 — Fig. 2, Supplementary Figs. 1–2

Day-ahead price differences between states. For each state hub and a reference hub, the deposited tables store

- `A+` and `A-`, the summed positive and negative differences
- `G = A+ + A-`, the gross magnitude
- `D = (A+ - A-) / G`, the directional bias

`results/price_spread_vs_refs/` holds summer and winter against California, Texas, and New York. Fig. 2 is the summer map. Supplementary Fig. 2 is the winter map. Supplementary Fig. 1 is the directional-motivation panel.

```text
python experiments/E02/plot.py
```

That writes `figures/Fig2_interstate_LMP_phase_lag.png`, `figures/ExtData_Fig2_price_direction_magnitude_winter.png`, and `figures/ExtData_Fig2_directional_motivation.png`. The motivation panel recomputes example price pairs from the state day-ahead cache, which is not in this folder. Place each series at

```text
experiments/data/lmp_cache/USA/state_<ST>/{summer_2026-07,winter_2026-01}.json
```

Hub names are in `config/state_lmp_registry.yaml`. January and July 2026 are 744 hours each, starting at 00:00 UTC on the first day of the month. `model.py` is the metric definition used for the directional-motivation panel.
