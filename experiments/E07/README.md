# E07 — Fig. 7, Supplementary Figs. 10–11

Battery and hydropower arbitrage under continental summer prices from E06, plus ordinary-load versus data-center growth comparisons. Panel c of Fig. 7 and Supplementary Fig. 11 uses observed day-ahead LMPs for January and July 2026 from the local (gitignored) `experiments/data/lmp_cache` month slices.

```text
python experiments/E07/plot.py
```

That writes `figures/Fig7_battery_arbitrage.png`.

```text
python experiments/E07/plot_load_growth.py
```

That writes `figures/ExtData_Fig7_load_growth_battery_arbitrage.png` (Supplementary Fig. 10).

```text
python experiments/E07/plot_hydro.py
```

That writes `figures/ExtData_Fig7_hydro_arbitrage.png` (Supplementary Fig. 11).
