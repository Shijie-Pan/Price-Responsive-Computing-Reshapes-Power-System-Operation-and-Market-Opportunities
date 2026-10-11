# E05 — Fig. 5, Supplementary Figs. 7–9

Continental linear dispatch on `experiments/data/networks/usa/elec_s500_c134c.nc`. The open-loop cases are in `results/priority_168h`: `no_dc`, `baseline`, and `reallocation_slack_termination`, each for 168 h. The price-responsive column of Fig. 5 is the dispatch after six schedule–network–price updates in `results/closed_loop/iter_06`.

```text
python experiments/E05/plot.py
```

That writes `figures/Fig5_endogenous_lmp_callback.png` and `figures/ExtData_Fig5_usa_endogenous_maps_winter.png` (Supplementary Fig. 8).

Closed-loop iteration board from the deposited summary:

```text
python experiments/E05/plot_closed_loop.py
```

That writes `figures/ExtData_Fig5_usa_endogenous_closed_loop.png` (Supplementary Fig. 7).

Expanded load-propagation board (tables in `results/load_propagation/`):

```text
python experiments/E05/plot_load_propagation.py
```

That writes `figures/ExtData_Fig5_load_propagation_full_flex.png` (Supplementary Fig. 9). The top row is the July 2025 background. The bottom row keeps the same hours after the 2030 data-center increment is scaled by 1.520461/0.607868, the 2050-to-2030 ratio for commercial data-center server electricity in EIA AEO2026 Table 5 (April 2026, Counterfactual Baseline).

The site list is `config/nationwide_operators.json`. An open-loop solve that matches the deposited window is:

```text
python experiments/E05/run_lopf.py --season both --eval-length 168 --tag priority_168h
```

The default tag is `priority_24h` and the default length is 24 h, which is not the manuscript window. The command above replaces `results/priority_168h`.

One closed-loop round, for summer, is:

```text
python experiments/E05/run_closed_loop.py export-lmp --tag priority_168h --season summer --iter 0
python experiments/E05/run_closed_loop.py e01 --season summer --iter 1
python experiments/E05/run_closed_loop.py lopf --season summer --iter 1
```

Repeat `e01` and `lopf` with `--iter` from 2 through 6. `e01` runs the E03 job model for every operator under the previous prices, so it needs the job files and a Gurobi licence. The deposited figure uses iteration 6.
