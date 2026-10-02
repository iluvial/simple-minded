# SST / DHW (Caribbean) and Hs (Tumaco) analysis — ERA5

- `sst_hs_analysis.ipynb` — the notebook (open in Colab/Jupyter, put the ERA5 files next to it, run all).
- `sst_hs_analysis.py` — the same notebook in "percent" format (readable diffs; `jupytext` can convert it back).

Inputs expected in the notebook folder: `2000.nc … 2020.nc` (SST, variable `sst`),
`Hs_2000-2012_Tumaco.nc`, `Hs_2013-2020_Tumaco.nc` (variable `swh`) and, optionally,
`oni.ascii.txt` (NOAA CPC ONI; downloaded automatically when internet is available).
