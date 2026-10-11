# Price-Responsive Computing Reshapes Power-System Operation and Market Opportunities

Code, configuration, and deposited results for the manuscript experiments. E01–E07 correspond to Figs. 1–7. Each experiment folder has a README with the redraw and recompute commands.

```text
python -m pip install -r requirements.txt
```

Recomputing a schedule or a network dispatch needs a local Gurobi licence. `gurobipy` must match that installation.

Redraw the deposited figures from this folder:

```text
python experiments/E01/plot_demand_maps.py --case background --season summer
python experiments/E02/plot.py
python experiments/E03/plot.py
python experiments/E03/plot_service_condition.py
python experiments/E03/plot_campus_map.py
python experiments/E04/plot.py
python experiments/E04/plot_ramp_limit.py
python experiments/E05/plot.py
python experiments/E05/plot_closed_loop.py
python experiments/E05/plot_load_propagation.py
python experiments/E06/plot.py
python experiments/E06/plot_hydro.py
python experiments/E06/plot_load_growth.py
python experiments/E07/plot.py
```

| Manuscript | Result |
|---|---|
| Fig. 1 | `experiments/E01/figures/Fig1_summer_peak_load.png` |
| Fig. 2 | `experiments/E02/figures/Fig2_interstate_LMP_phase_lag.png` |
| Fig. 3 | `experiments/E03/figures/Fig3_job_level_flexibility.png` |
| Fig. 4 | `experiments/E04/figures/Fig4_congestion_relocation.png` |
| Fig. 5 | `experiments/E05/figures/Fig5_endogenous_lmp_callback.png` |
| Fig. 6 | `experiments/E06/figures/Fig6_battery_arbitrage.png` |
| Fig. 7 | `experiments/E07/figures/Fig7_aggregate_model_error.png` |
| Supplementary Fig. 1 | `experiments/E02/figures/ExtData_Fig2_directional_motivation.png` |
| Supplementary Fig. 2 | `experiments/E02/figures/ExtData_Fig2_price_direction_magnitude_winter.png` |
| Supplementary Fig. 3 | `experiments/E03/figures/ExtData_Fig3_texas_campus_map.png` |
| Supplementary Fig. 4 | `experiments/E04/figures/ExtData_Fig4_ramp_limit_sensitivity.png` |
| Supplementary Fig. 5 | `experiments/E03/figures/ExtData_Fig3_service_condition_cost.png` |
| Supplementary Fig. 6 | `experiments/E03/figures/ExtData_Fig3_texas_site_trajectories.png` |
| Supplementary Fig. 7 | `experiments/E05/figures/ExtData_Fig5_usa_endogenous_closed_loop.png` |
| Supplementary Fig. 8 | `experiments/E05/figures/ExtData_Fig5_usa_endogenous_maps_winter.png` |
| Supplementary Fig. 9 | `experiments/E05/figures/ExtData_Fig5_load_propagation_full_flex.png` |
| Supplementary Fig. 10 | `experiments/E06/figures/ExtData_Fig6_load_growth_battery_arbitrage.png` |
| Supplementary Fig. 11 | `experiments/E06/figures/ExtData_Fig6_hydro_arbitrage.png` |

Fig. 5 uses `experiments/E05/results/closed_loop/iter_06` for the price-responsive case.
