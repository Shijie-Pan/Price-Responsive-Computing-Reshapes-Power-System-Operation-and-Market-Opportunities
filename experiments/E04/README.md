# E04 — Fig. 4, Supplementary Fig. 5

Texas transmission with the E03 schedules injected onto `experiments/data/networks/texas/elec_s1250.nc`. The three cases are `no_dc`, `baseline` (price-independent), and `reallocation_slack_termination` (price-responsive). The evaluation window is 168 h. Binding line-hours use a 99.9% loading threshold (same definition as Fig. 5). Winter uses the first week stored in the network. Summer overlays the July renewable week from the profile files.

Redraw Fig. 4 from the hourly line loadings in `results/flexibility/full/`:

```text
python experiments/E04/plot.py
```

Ramp-limit sensitivity (95% + 24 h cases) from deposited fleet and congestion tables:

```text
python experiments/E04/plot_ramp_limit.py
```

That writes `figures/ExtData_Fig4_ramp_limit_sensitivity.png` (Supplementary Fig. 5).

The network solve reads merged site power from `experiments/E08/merged_power`, as set in `config/pypsa_texas7.json`. Rebuild those two manuscript scenarios from the E03 power files:

```text
python experiments/E04/merge_power.py --scenario baseline
python experiments/E04/merge_power.py --scenario reallocation_slack_termination
```

That writes into `experiments/E08/merged_power` and replaces the deposited merged files. Then solve without replacing the deposited loadings:

```text
python experiments/E04/run_lopf.py --season both --priority --skip-voltage-pf --results-dir validation/e04
```

`--priority` without `--results-dir` writes into `results/flexibility/full`. The solver is Gurobi.
