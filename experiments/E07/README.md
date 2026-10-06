# E07 — Fig. 7

Comparison of the job-level schedule with three aggregate schedules on the same eleven Texas sites and 168 h: site energy (`AGG_ENERGY`), operator energy (`AGG_POOL`), and a continuous envelope (`AGG_CONT`). `config/comparison.yaml` sets the floor at 10 MW and the cap at 2200 MW. The formulation is `experiments/lib/solve/aggregate/scheduler.py`. `model.py` calls that formulation.

The deposited power series and `results/comparison_summary.json` are the figure inputs. Expenditure is price times facility power over 168 h. On these series, power stays at or above facility idle, so that product is the stored `energy_cost_usd`. Peak is the fleet peak. Ramp and interquartile range are for Amazon Pecos (site 751).

Redraw:

```text
python experiments/E07/plot.py
```

If the site price files from E03 are present, the expenditure bars are computed from them. If they are absent, the bars use the stored costs.

Solving again replaces `results/`:

```text
python experiments/E07/run_comparison.py --season both
```

`merged_power/` is the operator-level input used by the Texas network in E04, not a separate figure.
