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
python experiments/E06/plot_hydro.py
python experiments/E07/plot.py
python experiments/E08/plot.py
```

| Manuscript | Result |
|---|---|
| Fig. 1 | `experiments/E01/figures/Fig1_summer_peak_load.png` |
| Fig. 2 | `experiments/E02/figures/Fig2_interstate_LMP_phase_lag.png` |
| Fig. 3 | `experiments/E03/figures/Fig3_job_level_flexibility.png` |
| Fig. 4 | `experiments/E04/figures/Fig4_congestion_relocation.png` |
| Fig. 5 | `experiments/E05/figures/Fig5_endogenous_lmp_callback.png` |
| Fig. 6 | `experiments/E06/figures/Fig6_battery_arbitrage.png` |
| Fig. 7 | `experiments/E07/figures/Fig7_load_propagation_full_flex.png` |
| Fig. 8 | `experiments/E08/figures/Fig8_aggregate_model_error.png` |
| ExtData Fig. 2, with Fig. 2 | `experiments/E02/figures/ExtData_Fig2_directional_motivation.png` |
| ExtData Fig. 2, with Fig. 2 | `experiments/E02/figures/ExtData_Fig2_price_direction_magnitude_winter.png` |
| ExtData Fig. 3, with Fig. 3 | `experiments/E03/figures/ExtData_Fig3_texas_site_trajectories.png` |
| ExtData Fig. 4, with Fig. 4 | `experiments/E04/figures/ExtData_Fig4_texas_campus_map.png` |
| ExtData Fig. 5, with Fig. 5 | `experiments/E05/figures/ExtData_Fig5_usa_endogenous_maps_winter.png` |
| ExtData Fig. 6, with Fig. 6 | `experiments/E06/figures/ExtData_Fig6_hydro_arbitrage.png` |

Fig. 5 uses `experiments/E05/results/closed_loop/iter_06` for the price-responsive case.
