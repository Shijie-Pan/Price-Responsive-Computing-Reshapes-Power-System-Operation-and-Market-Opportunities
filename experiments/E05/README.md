# E05 — Fig. 5 and Supplementary Fig. 13

Continental linear dispatch on `experiments/data/networks/usa/elec_s500_c134c.nc`. The open-loop cases are in `results/priority_168h`: `no_dc`, `baseline`, and `reallocation_slack_termination`, each for 168 h. The price-responsive column of Fig. 5 is not that open-loop case. It is the dispatch after six schedule–network–price updates, in `results/closed_loop/iter_06`. Supplementary Fig. 13 is the winter open-loop map.

Redraw both figures:

```text
python experiments/E05/plot.py
```

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
