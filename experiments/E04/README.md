# E04 — Fig. 4 and Supplementary Fig. 11

Texas transmission with the E03 schedules injected onto `experiments/data/networks/texas/elec_s1250.nc`. The three cases are `no_dc`, `baseline` (price-independent), and `reallocation_slack_termination` (price-responsive). The evaluation window is 168 h. Winter uses the first week stored in the network. Summer overlays the July renewable week from the profile files.

Redraw Fig. 4 from the hourly line loadings in `results/flexibility/full/`:

```text
python experiments/E04/plot.py
```

Supplementary Fig. 11 is the site map:

```text
python experiments/E04/plot_campus_map.py
```

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
