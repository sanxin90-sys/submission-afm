# Data and code: thermal response time of air above and within Amazonian forest canopies

This package holds the processed data and the analysis code for the manuscript "Thermal response time of air above and within Amazonian forest canopies: seasonal, interannual and vertical variation", submitted to *Agricultural and Forest Meteorology*.

[AUTHORS, YEAR, REPOSITORY DOI: to be completed after deposit]

## Contents

```
code/    Python scripts (comments and docstrings in Portuguese)
data/    processed half-hourly series, fitted parameters, results tables and network weights
```

Not included:

- **Raw tower files.** Their owners are listed under "Data sources". `preprocess.py`, `jaru_anos.py` and `alturas.py` read them from `raw/` (or from the `RAW_DIR` environment variable).
- **Map layers for Fig. 1.** These are Natural Earth, the IBGE biomes and the Legal Amazon from geobr. `figuras_en.py` reads them from `geo/` (or `GEO`).

Every other script runs from `data/`.

## Requirements

Python 3 (tested with 3.13) with numpy, pandas, scipy, matplotlib, pillow, torch (only for the scripts that train or load the neural networks: `fit_nn_correction`, `bootstrap_nn`, `fit_ude`, `artigo1_aplicado`, `replica_artigo1`, `series_teste`, `simulacao_continua`), xlrd and openpyxl (raw files only), and geopandas (Fig. 1 only). See `requirements.txt`.

## Processed series (`data/*_30min.csv`)

All files have one row per 30-min interval, labeled at the end of the interval (`fim_intervalo`, local time). Gaps are left empty and nothing is gap-filled.

| Column | Meaning |
|---|---|
| `T_C`, `T_K` | Air temperature used in the model (°C, K): Jaru 62 m, Sinop 40 m, Cuieiras 51.1 m, Mujuí 12 m |
| `Rn` | Net radiation (W m⁻²) |
| `RH`, `q` | Relative humidity (%) and specific humidity (kg kg⁻¹) |
| `H`, `LE`, `G` | Sensible, latent and soil heat fluxes (W m⁻², Cuieiras only; eddy covariance at 53.1 m) |
| `Rsd`, `Rn_razao_dia` | Downward shortwave radiation and daily Rn/Rsd ratio (Sinop) |
| `T42_C`, `T28_C`, `T5_C`, `RH5`, `Ttr_*` | Additional levels and trunk temperatures |
| `nivel_rio_m`, `lamina_sitio_m`, `fase_hidro` | River level, water depth at the site and flood phase (Mujuí) |
| `estacao` | Season or flood phase of the main datasets (`chuvosa` = wet, `seca` = dry, `terrestre` = terrestrial, `inundada` = flooded) |
| `evento` | Flagged events (Mujuí cold-air incursion, 17–18 July 2010) |
| `ok_TR`, `ok_TRq` | Interval valid for T and Rn, or for T, Rn and q |

`jaru_anos_30min.csv` holds Jaru by year (2004–2010) and height (`altura`: 62m or 22m).

## Which script produces which result

Run the scripts from `code/`. Each writes its output to `data/`. `fit_dry_model.py` and `fit_baseline.py` are imported by most of the others.

| Script | Output in `data/` | Manuscript |
|---|---|---|
| `preprocess.py` (raw files) | `cuieiras_`, `jaru_`, `mujui_`, `sinop_30min.csv`, `segmentos.csv`, `resumo_janelas.csv`, `cobertura_mensal.csv` | Table 1 |
| `jaru_anos.py` (raw files) | `jaru_anos_30min.csv`, `jaru_anos_parametros.csv` | Sect. 3.4 |
| `fit_dry_model.py` | `modelo_seco_parametros.csv` | Sect. 3.3 |
| `artigo1_aplicado.py` | `palpites_iniciais.csv`, `multistart_linear.csv`, `artigo1_aplicado.csv` | Table 3, Table 5 |
| `replica_artigo1.py` | `replica_artigo1.csv` | Sect. 3.1, Fig. 2 |
| `fit_baseline.py` | `linha_base_parametros.csv` | Table 4 |
| `intervalo_predicao.py` | `intervalo_predicao_base.csv` | Table 4 (coverage) |
| `referencia_climatologica.py` | `referencia_climatologica.csv` | Table 5, Sect. 3.1 |
| `fit_nn_correction.py`, `bootstrap_nn.py` | `correcao_nn_resultados.csv`, `bootstrap_nn_resultados.csv` | Table 5, Sect. 3.3 |
| `fit_ude.py` | `ude_resultados.csv`, `ude_modelos/` | Table 5, Sect. 3.3 |
| `teq_variavel.py`, `teq_sensibilidade.py` | `teq_variavel.csv`, `teq_variavel_anos.csv`, `teq_variavel_sensibilidade.csv` | Table 5, Sects. 3.3–3.4 |
| `simulacao_continua.py` | `simulacao_continua.csv` | Sect. 3.3 |
| `series_teste.py` | `series_teste.csv`, `series_teste_janelas.csv` | Figs. 3–4 |
| `estacoes_anos.py` | `estacoes_anos_parametros.csv`, `meses_parametros.csv` | Figs. 5–6, Sect. 3.4 |
| `revisao_identificabilidade.py` | `perfil_tau.csv`, `revisao_identificabilidade.csv`, `janelas_sensibilidade.csv`, `tau_umidade_controle.csv` | Sects. 3.2 and 3.4 |
| `alturas.py` (raw files) | `alturas_parametros.csv` | Fig. 9, Sect. 3.5 |
| `fit_transfer.py` | `transferencia_base.csv` | Fig. 10, Sect. 3.6 |
| `fluxos_cuieiras.py` | `fluxos_cuieiras_janelas.csv`, `fluxos_cuieiras_resumo.csv` | Table 6, Fig. 7, Sect. 3.4 |
| `tau_dia_noite.py` (profiles need raw files) | `tau_dia_noite.csv` | Eq. 14, Fig. 8, Sects. 3.4–3.5 |
| `figuras_en.py` | `../figures/Fig1`–`Fig7`, `Fig_ciclo`, `Fig_fluxos` and `Fig_dianoite` (published as Figs. 1–3, 5, 6, 9 and 10, and 4, 7 and 8); `ciclo_diurno_teste_metricas.csv` (RMSE, MAE and R² in Fig. 4). Pass figure names to build only some, e.g. `python3 figuras_en.py series ciclo fluxos` | Figs. 1–10 |

Random seeds are fixed in each script. Bootstrap replicates and network training take from minutes to hours on a laptop. In a clean copy of this package, `fluxos_cuieiras.py` and `referencia_climatologica.py` reproduced the files in `data/`, and `figuras_en.py series ciclo fluxos` produced Figs. 3, 4 and 7. `fit_baseline.py` reproduced `linha_base_parametros.csv` (the two Cuieiras rows in this file come from that rerun; an earlier partial run had given slightly different bootstrap intervals for them). The other scripts were not rerun from the package. Network training may differ slightly across PyTorch versions or hardware.

## Data sources

[CONFIRM WITH EACH PROVIDER: the conditions under which the processed series can be redistributed, and the required acknowledgement.]

- **Jaru Biological Reserve (Rondônia):** tower data used in da Silva et al. (2026), Chaos 36:033134, https://doi.org/10.1063/5.0303533. [Provider: LBA Program]
- **Sinop (Mato Grosso):** tower data from de Paulo et al. (2015), Chaos 25:123123, https://doi.org/10.1063/1.4938188. [Provider: Sérgio de Paulo]
- **Cuieiras Reserve, K34 tower (Amazonas):** meteorological, radiation and eddy-covariance data. [Provider: LBA Program]
- **Mujuí, Mamirauá Sustainable Development Reserve (Amazonas):** tower and river-level data from da Silva (2019). [Provider: Alessandro Michiles]

## License

CC BY 4.0 for the data and MIT for the code, provided the data providers allow it.
