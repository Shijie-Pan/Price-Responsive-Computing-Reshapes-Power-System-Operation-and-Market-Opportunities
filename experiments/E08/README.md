# E08 — Fig. 8

The same regional-demand map as E01, with data-center demand added at the expanded summer capacity and then shifted by the price-responsive schedule. The hourly shapes are in `results/`:

- `hybrid_demand_hod_mean_summer_2025.csv`
- `hybrid_demand_hod_mean_summer_2025_with_dc.csv`
- `hybrid_demand_hod_mean_summer_2025_full_flex.csv`
- `hybrid_nodes_summer.csv`

From the deposit root:

```text
python experiments/E08/plot.py
```

That calls the E01 drawing code and writes `figures/Fig7_load_propagation_full_flex.png`. The underlying demand year is July 2025, as in E01.
