# E03 — Fig. 3, Supplementary Figs. 3–5

Job-level schedules for six operators and eleven Texas sites. `baseline` is price-independent. `reallocation_slack_termination` is price-responsive. The solve covers 7 days plus 24 hours (192 h). Demand profiles are evaluated over the original 168 hours. Electricity expenditure includes dynamic consumption over the 192-hour scheduling horizon and idle consumption over the original 168 hours.

`model.py` is the CPU allocation model. `config/texas7_operators.json` is the site list, capacity, PUE, and trace window. `manifest.json` names the power files used by the figures.

```text
python experiments/E03/plot.py
```

That writes `figures/Fig3_job_level_flexibility.png` and `figures/ExtData_Fig3_texas_site_trajectories.png` (Supplementary Fig. 5).

Service-condition costs (price-independent, 95% + 24 h, and 100% with no extension):

```text
python experiments/E03/plot_service_condition.py
```

That writes `figures/ExtData_Fig3_service_condition_cost.png` (Supplementary Fig. 4).

Texas campus map:

```text
python experiments/E03/plot_campus_map.py
```

That writes `figures/ExtData_Fig3_texas_campus_map.png` (Supplementary Fig. 3).

To solve again, two local inputs are required.

1. Download `job_info_df.csv` from the Alibaba spot-GPU cluster trace:

   https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-spot-gpu

   Save it as `experiments/data/sources/job_info_df.csv`, then:

   ```text
   python experiments/E03/prepare_jobs.py
   ```

   The script keeps `gpu_model == "A10"` and writes one job file per window under `experiments/data/jobs/by_trace/`. Each operator uses the `trace_segment` in the config: CoreWeave `days_121_151`, Oracle `days_75_105`, Google `days_155_185`, Meta `days_30_60`, Microsoft `days_60_90`, Amazon `days_90_120`.

2. Save ERCOT day-ahead settlement-point prices for January 2026 and July 2026 as

   ```text
   experiments/data/lmp_cache/ERCOT/nodal/<node>/winter_2026-01.json
   experiments/data/lmp_cache/ERCOT/nodal/<node>/summer_2026-07.json
   ```

   Each file has a `prices` list of 744 hourly values. Then:

   ```text
   python experiments/E03/prepare_site_lmp.py
   ```

One operator and season, written beside the deposited results rather than over them:

```text
python experiments/E03/run_schedule.py --operator amazon --season summer --scenarios baseline reallocation_slack_termination --results validation/e03_amazon_summer
```

Omit `--results` to write under `results/<operator>/`. The manuscript gap is 0.002 and the time limit is 28800 seconds. Google uses a gap of 0.005. `--smoke` is a short check, not the manuscript solve.
