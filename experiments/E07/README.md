# E07 — Fig. 7

The same regional-demand map as E01, with data-center demand added at the expanded summer capacity and then shifted by the price-responsive schedule. The hourly shapes are in `results/`:

- `hybrid_demand_hod_mean_summer_2025.csv`
- `hybrid_demand_hod_mean_summer_2025_with_dc.csv`
- `hybrid_demand_hod_mean_summer_2025_full_flex.csv`
- `hybrid_nodes_summer.csv`

From the deposit root:

```text
python experiments/E07/plot.py
```

That calls the E01 drawing code and writes `figures/Fig7_load_propagation_full_flex.png`. The top row is the July 2025 background. The bottom row keeps the same hours after the 2030 data-center increment is scaled by 1.520461/0.607868, the 2050-to-2030 ratio for commercial data-center server electricity in EIA AEO2026 Table 5 (April 2026, Counterfactual Baseline).
