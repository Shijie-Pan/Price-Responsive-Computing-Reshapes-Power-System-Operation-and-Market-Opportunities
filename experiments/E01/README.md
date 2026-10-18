# E01 — Fig. 1

Regional demand relative to each location's own peak. The series are hourly EIA-930 and FERC-714 demand for January and July 2025 in `experiments/data/eia930/`. This experiment does not solve a dispatch model.

From the deposit root:

```text
python experiments/E01/plot_demand_maps.py --case background --season summer
```

That writes `figures/Fig1_summer_peak_load.png`. The same script draws the other demand boards. Supplementary Fig. 9 (load propagation) is produced from E06.
