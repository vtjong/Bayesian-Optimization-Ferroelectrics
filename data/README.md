# Data

Every file here is either a measured input the code reads or a record the campaign produced.
Nothing in `src/` mirrors these numbers as literals; the paths live in `src/paths.py`.

| File | What it is | Read by |
| --- | --- | --- |
| `flash_temp_table.csv` | Simulated peak temperature `Tmax(V, t)` of the PulseForge lamp on the 30-node grid (CERA model). Defines the design box. | `paths.FLASH_TABLE_CSV` |
| `flash_temp_table_skr.csv` | Second, independent simulation of the same tool on the same nodes; used only to bound the disagreement between thermal models. | `paths.FLASH_TABLE_SKR_CSV` |
| `Bolometer_readings_PulseForge.xlsx`, `Bolometer_readings_PulseForge_new.xlsx` | Delivered fluence `E(V, t)` measured with the bolometer; the lamp irradiance envelope in `physics/constants.py` was inverted from these. | documentation only |
| `flash_plan_seed.csv`, `flash_plan_seed_results.csv`, `campaign_seed.csv`, `campaign_seed_results.csv` | The fired seed plan and the sheet the laboratory fills in. | `src/run_initial_design*.py` |
| `electrical/` | Raw Keithley 4200A exports per round (`raw/round<N>/Sample<NN>/`), the group's reference summaries (`reference/`), and the feature tables the electrical pipeline writes (`features/`). | `src/electrical/` |

The 2023 archival exports of the earlier KHM005-010 samples (a different tool and film stack)
were removed from the tree on 2026-10-23; they remain in git history before that commit.
