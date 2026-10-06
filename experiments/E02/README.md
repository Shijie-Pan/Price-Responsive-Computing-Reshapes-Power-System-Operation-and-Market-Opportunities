# E02 — Fig. 2 and Supplementary Figs. 9 and 10

Day-ahead price differences between states. For each state hub and a reference hub, the deposited tables store

- `A+` and `A-`, the summed positive and negative differences
- `G = A+ + A-`, the gross magnitude
- `D = (A+ - A-) / G`, the directional bias

`results/price_spread_vs_refs/` holds summer and winter against California, Texas, and New York. Fig. 2 is the summer map. Supplementary Fig. 10 is the winter map. Both are drawn from those tables:

```text
python experiments/E02/plot.py
```

The command also builds Supplementary Fig. 9. That panel recomputes example price pairs from the state day-ahead cache, which is not in this folder. Place each series at

```text
experiments/data/lmp_cache/USA/state_<ST>/{summer_2026-07,winter_2026-01}.json
```

Hub names are in `config/state_lmp_registry.yaml`. January and July 2026 are 744 hours each, starting at 00:00 UTC on the first day of the month. `model.py` is the metric definition used for Supplementary Fig. 9.
