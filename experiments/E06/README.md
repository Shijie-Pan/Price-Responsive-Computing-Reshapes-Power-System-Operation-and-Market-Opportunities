# E06 — Fig. 6, Supplementary Figs. 8–10

Continental linear dispatch on `experiments/data/networks/usa/elec_s500_c134c.nc`. Open-loop cases are in `results/priority_168h`: `no_dc`, `baseline`, and `reallocation_slack_termination` (168 h). The price-responsive column of Fig. 6 uses `results/closed_loop/iter_06` after six schedule–network–price updates.

```text
python experiments/E06/plot.py
```

That writes `figures/Fig6_endogenous_lmp_callback.png` and `figures/ExtData_Fig6_usa_endogenous_maps_winter.png` (Supplementary Fig. 9).

Closed-loop iteration board:

```text
python experiments/E06/plot_closed_loop.py
```

That writes `figures/ExtData_Fig6_usa_endogenous_closed_loop.png` (Supplementary Fig. 8).

Load-propagation board (`results/load_propagation/`):

```text
python experiments/E06/plot_load_propagation.py
```

That writes `figures/ExtData_Fig6_load_propagation_full_flex.png` (Supplementary Fig. 10). The top row is the July 2025 background. The bottom row uses the same hours after scaling the 2030 data-center increment by 1.520461/0.607868 (EIA AEO2026 Table 5, April 2026, Counterfactual Baseline, 2050-to-2030 ratio for commercial data-center server electricity).

Site list: `config/nationwide_operators.json`. Open-loop solve matching the deposited window:

```text
python experiments/E06/run_lopf.py --season both --eval-length 168 --tag priority_168h
```

One closed-loop round (summer):

```text
python experiments/E06/run_closed_loop.py export-lmp --tag priority_168h --season summer --iter 0
python experiments/E06/run_closed_loop.py e01 --season summer --iter 1
python experiments/E06/run_closed_loop.py lopf --season summer --iter 1
```
