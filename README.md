# Price-Responsive Computing Reshapes Power-System Operation and Market Opportunities

Code, configuration, and deposited results for the manuscript experiments. E01–E08 correspond to Figs. 1–8. Each experiment folder has a README with the redraw and recompute commands.

```text
python -m pip install -r requirements.txt
```

Recomputing a schedule or a network dispatch needs a local Gurobi licence. `gurobipy` must match that installation.

Redraw the deposited figures from this folder:

```text
python experiments/E01/plot_demand_maps.py --case background --season summer
python experiments/E02/plot.py
python experiments/E03/plot.py
python experiments/E04/plot.py
python experiments/E04/plot_campus_map.py
python experiments/E05/plot.py
python experiments/E06/plot.py
python experiments/E07/plot.py
python experiments/E08/plot.py
```

| Manuscript | Result |
|---|---|
| Fig. 1 | `experiments/E01/figures/Fig_intro_summer_peak_load.png` |
| Fig. 2 | `experiments/E02/figures/Fig1_interstate_LMP_phase_lag.png` |
| Fig. 3 | `experiments/E03/figures/Fig2_job_level_flexibility.png` |
| Fig. 4 | `experiments/E04/figures/Fig3_congestion_relocation.png` |
| Fig. 5 | `experiments/E05/figures/Fig4_endogenous_lmp_callback.png` |
| Fig. 6 | `experiments/E06/figures/Fig5_battery_arbitrage.png` |
| Fig. 7 | `experiments/E07/figures/Fig6_aggregate_model_error.png` |
| Fig. 8 | `experiments/E08/figures/Fig7_load_propagation_full_flex.png` |
| Supplementary Fig. 9 | `experiments/E02/figures/ExtData_Fig6_directional_motivation.png` |
| Supplementary Fig. 10 | `experiments/E02/figures/ExtData_Fig1_price_direction_magnitude_winter.png` |
| Supplementary Fig. 11 | `experiments/E04/figures/ExtData_Fig4_texas_campus_map.png` |
| Supplementary Fig. 12 | `experiments/E03/figures/ExtData_Fig_texas_site_trajectories.png` |
| Supplementary Fig. 13 | `experiments/E05/figures/ExtData_Fig_usa_endogenous_maps_winter.png` |

Fig. 5 uses `experiments/E05/results/closed_loop/iter_06` for the price-responsive case.
