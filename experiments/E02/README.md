# E02 — Fig. 2, Supplementary Fig. 1

Day-ahead price differences between states. For each state hub and a reference hub, the deposited tables store

- `A+` and `A-`, the summed positive and negative differences
- `G = A+ + A-`, the gross magnitude
- `D = (A+ - A-) / G`, the directional bias

`results/price_spread_vs_refs/` holds summer and winter against California, Texas, and New York. Fig. 2 is the summer map. Supplementary Fig. 1 is the winter map. `model.py` defines `D` and `G` for those deposited tables.

```text
python experiments/E02/plot.py
```

That writes `figures/Fig2_interstate_LMP_phase_lag.png` and `figures/ExtData_Fig2_price_direction_magnitude_winter.png`.

Hub names are in `config/state_lmp_registry.yaml`. January and July 2026 are 744 hours each, starting at 00:00 UTC on the first day of the month.
