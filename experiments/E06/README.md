# E06 — Fig. 6

A 1 MW, 4 MWh battery at each bus. Round-trip efficiency is 0.9. The state of charge starts and ends at 2 MWh. Charge and discharge do not occur in the same hour. The battery takes the 168 h price as given.

Panels a and b compare the open-loop price-independent duals with the last price-responsive iteration. The weekly revenue and intraday range are already in `results/summer_battery_vs_price_independent.csv`.

Panel c applies the same battery to observed day-ahead prices and scales the price-responsive revenue by the change in intraday range. Capital cost is 326 USD/kWh in 2022 dollars for a 4-hour system (Cole and Karmakar, NREL/TP-6A40-85332). That panel reads:

```text
experiments/data/lmp_cache/ERCOT/LZ_NORTH/winter_2026-01.json
experiments/data/lmp_cache/ERCOT/LZ_NORTH/summer_2026-07.json
experiments/data/lmp_cache/MISO/MICHIGAN.HUB/annual_2025-08_2026-07.json
experiments/data/lmp_cache/CAISO/SP15/annual_2025-08_2026-07.json
```

The annual window is 1 August 2025 through 31 July 2026, 8760 hours, starting at 00:00 UTC. With those files in place:

```text
python experiments/E06/plot.py
```

This writes `figures/Fig5_battery_arbitrage.png`. The weekly table is not a substitute for the payback panel: the table uses model duals, and the payback panel uses the observed annual prices.
