# E05 — Fig. 5, Supplementary Fig. 6

Four Texas scheduling configurations under summer prices (exogenous day-ahead LMP), and the ordinary-load demand-response comparison referenced next to Fig. 5.

Redraw Fig. 5 from deposited line loadings in `results/four_mode/`:

```text
python experiments/E05/plot_four_mode.py
```

That writes `figures/Fig5_four_mode_dc_summer.png`.

Redraw Supplementary Fig. 6 from deposited loadings in `results/ordinary_load_dr/`:

```text
python experiments/E05/plot_ordinary_dr.py
```

That writes `figures/ExtData_Fig5_ordinary_load_dr_summer.png`. Map hour matches the Full-flex peak used for Fig. 4. Site markers appear only on the data-center panels.

Deposited files are redraw inputs only (hourly line loading and line geometry). Mechanism diagnostics are not included.
