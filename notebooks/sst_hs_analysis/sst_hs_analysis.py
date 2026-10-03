# %% [markdown]
# # Coral thermal stress in the Caribbean and wave climate at Tumaco (ERA5)
#
# **Part A – Sea-surface temperature (SST), Caribbean Sea**
#
# | Task | Section |
# |---|---|
# | 1. Monthly mean SST for every year of the record | A1 |
# | 2. Climatological averages | A2 |
# | 3. Empirical probability (PDF) and cumulative probability (CDF) distributions | A3 |
# | 4. Degree Heating Weeks (DHW) for two years, from daily mean SST | A4 |
#
# **Part B – Significant wave height (Hs), Tumaco (Colombian Pacific)**
#
# | Task | Section |
# |---|---|
# | 11. Daily mean Hs for every year | B1 |
# | 12. Monthly mean Hs for every year | B2 |
# | 13. Annual increase (trend) of Hs and its relation to climate change | B3 |
# | 14. Detrended Hs, climatology, anomaly series | B4 |
# | 15. FFT and Welch periodogram of 20 years of monthly Hs | B5 |
# | 16. The same for the Hs anomaly, and the two compared | B6 |
# | 17. Hs generated only by ENSO frequencies (2–4 yr band-pass) | B7 |
#
# **How to run it.** Put the ERA5 files next to the notebook (yearly SST files `2000.nc … 2020.nc`
# and the two Tumaco Hs files), adjust the configuration cell if needed, and run all cells.
# The first run reduces the hourly / 6-hourly files to **daily means** once and caches them
# (`sst_caribbean_daily.nc`, `hs_tumaco_daily.nc`). Every task in this notebook works with
# daily or monthly means, so this cuts memory use and run time by a factor of 4–24 without
# changing any result. Figures are shown inline and saved in `figures/`.

# %% [markdown]
# ## 0. Setup and configuration

# %%
import re
import urllib.request
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from scipy import signal, stats

# ---------------------------------------------------------------- data files --
DATA_DIR = Path(".")
FIG_DIR = Path("figures")

SST_VAR = "sst"
SST_YEARLY_PATTERN = r"^\d{4}\.nc$"               # 2000.nc, 2001.nc, ... only
SST_LEGACY_COMBINED = DATA_DIR / "sst_caribbean_2000_2021.nc"  # used if no yearly files
SST_DAILY_FILE = DATA_DIR / "sst_caribbean_daily.nc"            # cache written here

HS_VAR = "swh"                                     # ERA5 combined wind-sea + swell Hs
HS_SOURCES = [DATA_DIR / "Hs_2000-2012_Tumaco.nc", DATA_DIR / "Hs_2013-2020_Tumaco.nc"]
HS_LEGACY_COMBINED = DATA_DIR / "Hs_2000-2020_Tumaco.nc"
HS_DAILY_FILE = DATA_DIR / "hs_tumaco_daily.nc"

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_FILE = DATA_DIR / "oni.ascii.txt"

# ------------------------------------------------------- Part A: SST / DHW --
# Any two years of the record. 2005 and 2010 are the two documented Caribbean
# mass-bleaching years of this period (Eakin et al. 2010; Alemu & Clement 2014);
# figure A4e shows every year so the choice can be checked against the data.
DHW_YEARS = (2005, 2010)
SST_SITE = None            # (lat, lon) of a reef site; None = most stressed grid cell
MMM_BASELINE = None        # (first_year, last_year); None = whole record

DHW_WINDOW_DAYS = 84       # 12-week accumulation window
HOTSPOT_MIN = 1.0          # only HotSpots >= 1 degC accumulate
ALERT1, ALERT2 = 4.0, 8.0  # degC-weeks: bleaching likely / widespread bleaching + mortality

# ---------------------------------------------------------- Part B: waves ---
HS_SITE = (2.0, -79.0)     # (lat, lon) off Tumaco; None = area mean of the ocean cells
SPECTRAL_YEARS = 20        # task 15: 20 years of monthly data
WELCH_SEGMENT_YEARS = 8    # 8-yr Hann segments, 50 % overlap -> 4 segments in 20 yr
ENSO_BAND = (2.0, 4.0)     # band-pass limits as periods (years)
FILTER_ORDER = 2           # Butterworth order (doubled by the forward-backward pass)
FS = 12.0                  # monthly data: 12 samples per year

# ------------------------------------------------------------------ style ---
FIG_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 160,
                     "axes.titleweight": "bold", "axes.edgecolor": "#444444"})
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REF_YEAR = 2001            # non-leap year used as a common seasonal axis
DARK, LINE = "#1f3b6f", "#1f5f8b"
POS, NEG = "#c44e52", "#4c72b0"
ANOM_C, BAND_C = "#e07b39", "#fde2b8"
NINO_BG, NINA_BG = "#fbe0dc", "#dce8f5"

# %% [markdown]
# ### Shared helpers
# Data access, statistics and plotting helpers used by both parts. Keeping them in one place
# means both datasets are read, averaged and tested in the same way.

# %%
# ===================================================================== I/O ===
def standardise(ds):
    """Common layout: time dim called 'time', no ERA5 bookkeeping coords,
    longitudes in -180..180, every coordinate ascending."""
    tdim = next((d for d in ("valid_time", "time") if d in ds.dims), None)
    if tdim is None:
        raise KeyError(f"No time dimension found; dims are {list(ds.dims)}")
    if tdim != "time":
        ds = ds.rename({tdim: "time"})
    ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])
    if float(ds["longitude"].max()) > 180:
        ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180)
    return ds.sortby(["time", "latitude", "longitude"])


def to_physical_units(da):
    """ERA5 stores SST in kelvin; return degC. Other variables pass through."""
    units = str(da.attrs.get("units", "")).lower()
    if units in ("k", "kelvin") or (da.name == "sst" and float(da.max()) > 200):
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da


def _daily_sum_count(path, var):
    """Per-day sum and sample count of one file (exact daily means can then
    be formed even if a day is split across two files)."""
    with xr.open_dataset(path) as ds:
        da = to_physical_units(standardise(ds)[var].load())
    step = pd.Series(da["time"].values).diff().median()
    s = da.resample(time="1D").sum(min_count=1)
    c = da.notnull().resample(time="1D").sum()
    return s, c, step


def build_daily_file(paths, out, var):
    """Combine source files in time and reduce them to daily means.

    Checks that every file has the same grid, merges overlapping timestamps,
    and reports missing days, so problems are visible instead of silent."""
    sums, counts, ref = [], [], None
    for p in paths:
        s, c, step = _daily_sum_count(p, var)
        grid = (s["latitude"].values, s["longitude"].values)
        if ref is None:
            ref = grid
        elif not all(np.array_equal(a, b) for a, b in zip(grid, ref)):
            raise ValueError(f"{p.name}: grid differs from {paths[0].name}; "
                             "files cover different areas and cannot be stacked.")
        print(f"  {p.name:<28} {pd.Timestamp(s.time.values[0]):%Y-%m-%d} to "
              f"{pd.Timestamp(s.time.values[-1]):%Y-%m-%d}, time step {step}")
        sums.append(s)
        counts.append(c)

    t = np.concatenate([s["time"].values for s in sums])
    days, inv = np.unique(t, return_inverse=True)
    shape = (len(days),) + sums[0].shape[1:]
    S, C = np.zeros(shape), np.zeros(shape)
    np.add.at(S, inv, np.nan_to_num(np.concatenate([s.values for s in sums])))
    np.add.at(C, inv, np.concatenate([c.values for c in counts]))
    daily = np.where(C > 0, S / np.where(C > 0, C, 1), np.nan)

    full = pd.date_range(days[0], days[-1], freq="D")
    print(f"  -> {len(days)} days; {len(full) - len(days)} missing days; "
          f"{len(t) - len(days)} days found in more than one file (merged)")
    da = xr.DataArray(daily.astype("float32"), name=var,
                      dims=("time", "latitude", "longitude"),
                      coords={"time": days, "latitude": ref[0], "longitude": ref[1]},
                      attrs={"units": sums[0].attrs.get("units", ""),
                             "history": "daily means of " + ", ".join(p.name for p in paths)})
    da.to_dataset().to_netcdf(out, encoding={var: {"zlib": True, "complevel": 4}})
    print(f"  saved {out}")


def prepare_daily(out, var, sources, legacy):
    """Use the cached daily file if present; otherwise build it from the
    source files (preferred) or from a previously combined file."""
    if out.exists():
        print(f"Using cached daily file {out}")
        return
    if sources and all(p.exists() for p in sources):
        print(f"Building {out} from {len(sources)} files:")
        build_daily_file(sources, out, var)
    elif legacy.exists():
        print(f"Building {out} from {legacy}:")
        build_daily_file([legacy], out, var)
    else:
        raise FileNotFoundError(f"No input for '{var}': expected {[p.name for p in sources]}"
                                f" or {legacy.name} in {DATA_DIR.resolve()}")


def open_daily(path, var):
    """Daily field (time, latitude, longitude) on a gap-free daily axis."""
    with xr.open_dataset(path) as ds:
        da = standardise(ds)[var].load().astype("float64")
    full = pd.date_range(da["time"].values[0], da["time"].values[-1], freq="D")
    if len(full) != da.sizes["time"]:
        print(f"  NOTE: {len(full) - da.sizes['time']} missing days set to NaN")
        da = da.reindex(time=full)
    return da


# ============================================================== geography ===
def fmt_coord(lat, lon):
    return (f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}")


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def ocean_mask(da):
    return da.notnull().any("time")


def area_weights(da):
    """cos(latitude): each cell weighted by the surface area it represents."""
    return np.cos(np.deg2rad(da["latitude"]))


def area_mean(da):
    """Area-weighted mean over the ocean cells (land NaN is excluded, not zero)."""
    s = da.weighted(area_weights(da)).mean(("latitude", "longitude")).to_series()
    s.index = pd.DatetimeIndex(s.index)
    return s


def area_fraction(mask, ocean):
    """Area-weighted fraction of the ocean where `mask` is True."""
    w = area_weights(mask).broadcast_like(ocean).where(ocean)
    return (mask.where(ocean) * w).sum(("latitude", "longitude")) / w.sum()


def nearest_ocean_cell(da, lat, lon):
    """Grid cell nearest to (lat, lon) by great-circle distance that holds data."""
    lats, lons = da["latitude"].values, da["longitude"].values
    if not (lats.min() - 1 <= lat <= lats.max() + 1 and lons.min() - 1 <= lon <= lons.max() + 1):
        raise ValueError(f"{fmt_coord(lat, lon)} is outside the data domain")
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    dist = haversine_km(lat, lon, LAT, LON)
    ocean = ocean_mask(da).transpose("latitude", "longitude").values
    i, j = np.unravel_index(np.argmin(np.where(ocean, dist, np.inf)), dist.shape)
    return {"lat": float(lats[i]), "lon": float(lons[j]), "dist_km": float(dist[i, j]),
            "nearest_is_land": not bool(ocean.flat[np.argmin(dist)])}


def cell_series(da, cell):
    s = da.sel(latitude=cell["lat"], longitude=cell["lon"]).to_series()
    s.index = pd.DatetimeIndex(s.index)
    return s


# ============================================================ time series ===
def monthly_mean(daily, min_days=20):
    """Monthly mean of daily means (every day weighs the same); months with
    fewer than `min_days` valid days are left as NaN."""
    g = daily.resample("MS")
    return g.mean().where(g.count() >= min_days)


def annual_mean(daily, min_days=330):
    g = daily.groupby(daily.index.year)
    return g.mean()[g.count() >= min_days]


def year_month_grid(monthly):
    grid = (monthly.to_frame("v").assign(year=monthly.index.year, month=monthly.index.month)
            .pivot_table(index="year", columns="month", values="v")
            .reindex(columns=range(1, 13)))
    grid.columns = MONTHS
    return grid


def doy365(index):
    """Day of year 1..365 with 29 Feb merged into 28 Feb, so years align."""
    idx = pd.DatetimeIndex(index)
    return np.asarray(idx.dayofyear - ((idx.is_leap_year) & (idx.dayofyear > 59)))


def seasonal_axis(index):
    """Dates of any year mapped onto REF_YEAR, to overlay several years."""
    return pd.Timestamp(REF_YEAR, 1, 1) + pd.to_timedelta(doy365(index) - 1, unit="D")


def decimal_year(index, kind):
    """Decimal year at the centre of each averaging period."""
    idx = pd.DatetimeIndex(index)
    mid = (idx + pd.Timedelta(hours=12) if kind == "daily"
           else idx + pd.to_timedelta(np.asarray(idx.days_in_month) / 2.0, unit="D"))
    start = pd.to_datetime(pd.DataFrame({"year": mid.year, "month": 1, "day": 1}))
    length = np.where(mid.is_leap_year, 366.0, 365.0)
    frac = np.asarray((mid - pd.DatetimeIndex(start)).total_seconds()) / 86400 / length
    return np.asarray(mid.year, float) + frac


def mid_month(index):
    return pd.DatetimeIndex(index) + pd.Timedelta(days=14)


# ============================================================== statistics ===
def lag1(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def n_eff_ar1(n, r1):
    """Effective sample size of an AR(1) series (Bretherton et al. 1999)."""
    r1 = max(r1, 0.0)
    return n * (1 - r1) / (1 + r1)


def harmonic_trend(series, kind, n_harm=3):
    """Least-squares fit of  y = a + b (t - t_mean) + sum_k [c_k cos 2pi k t + d_k sin 2pi k t]
    (t in decimal years). The harmonics absorb the seasonal cycle, so b is the
    trend (units per year) and is not biased by where the record starts and
    ends in the seasonal cycle. Standard error and degrees of freedom use the
    effective sample size of the autocorrelated residuals (Santer et al. 2000)."""
    t_all = decimal_year(series.index, kind)
    y = series.values.astype(float)
    ok = np.isfinite(y)
    y, t = y[ok], t_all[ok]
    tm = t.mean()
    cols = [np.ones_like(t), t - tm]
    for k in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    r1 = lag1(resid)
    dof = max(n_eff_ar1(n, r1) - p, 1.0)
    se = float(np.sqrt(resid @ resid / dof * np.linalg.inv(X.T @ X)[1, 1]))
    b = float(beta[1])
    tc = stats.t.ppf(0.975, dof)
    return {"slope": b, "se": se, "ci": (b - tc * se, b + tc * se),
            "p": float(2 * stats.t.sf(abs(b / se), dof)), "r1": r1, "n": n,
            "n_eff": n_eff_ar1(n, r1), "level": float(beta[0]), "t_mean": tm,
            "mean": float(y.mean()), "fit": beta[0] + b * (t_all - tm)}


def sen_mk(annual):
    """Sen's slope (95 % CI), Mann-Kendall test and least squares on annual
    values. Rank-based, so one extreme year cannot drive the result."""
    x, y = annual.index.values.astype(float), annual.values.astype(float)
    sen = stats.theilslopes(y, x)
    ols = stats.linregress(x, y)
    mk = stats.kendalltau(x, y)
    return {"sen": sen[0], "sen_int": sen[1], "sen_lo": sen[2], "sen_hi": sen[3],
            "tau": mk[0], "mk_p": mk[1], "ols": ols.slope, "ols_p": ols.pvalue,
            "n": len(y), "x": x, "y": y}


# ================================================================ plotting ===
def finish(fig, name, note=None):
    """Footer with the data source/location, save to FIG_DIR, show inline."""
    if note:
        fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=9, color="#555555")
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def text_box(ax, text, loc="upper right", fontsize=9.5, family=None):
    x, ha = (0.985, "right") if "right" in loc else (0.015, "left")
    y, va = (0.965, "top") if "upper" in loc else (0.035, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            family=family, linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc", alpha=0.95))


def month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp(REF_YEAR, 1, 1), pd.Timestamp(REF_YEAR, 12, 31))
    ax.set_xlabel("")


def year_axis(ax, step=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("")


def map_axes(ax, da):
    ax.set_facecolor("#c9c2b6")                    # land / no data
    ax.set_aspect(1 / np.cos(np.deg2rad(float(da["latitude"].mean()))))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)


def fmt_period(years):
    return f"{years:.1f} yr" if years >= 1 else f"{years * 12:.1f} mo"


# %% [markdown]
# ## Data preparation
# The SST cell of the previous version selected files with `glob("*.nc")`, which also picked up
# the `Hs_*.nc` files and crashed (`int('Hs_2013-2020_Tumaco')`). Files are now matched by an
# explicit pattern, the grids are checked to be identical, and overlapping or missing days are
# reported.

# %%
sst_sources = sorted(p for p in DATA_DIR.iterdir() if re.match(SST_YEARLY_PATTERN, p.name))
prepare_daily(SST_DAILY_FILE, SST_VAR, sst_sources, SST_LEGACY_COMBINED)
prepare_daily(HS_DAILY_FILE, HS_VAR, HS_SOURCES, HS_LEGACY_COMBINED)

# %% [markdown]
# ---
# # Part A – Sea-surface temperature and coral thermal stress (Caribbean Sea)
#
# Tasks 1–3 describe the SST of the **whole domain** (area-weighted mean over ocean cells, so
# each cell counts in proportion to its surface area and land is excluded). Task 4 (DHW) is
# computed **cell by cell**, because thermal stress is a local quantity (see A4).

# %%
sst = open_daily(SST_DAILY_FILE, SST_VAR)
sst_ocean = ocean_mask(sst)
sst_area = area_mean(sst)
sst_month = monthly_mean(sst_area)
lat_rng = (float(sst.latitude.min()), float(sst.latitude.max()))
lon_rng = (float(sst.longitude.min()), float(sst.longitude.max()))
SST_PLACE = (f"ERA5 SST, Caribbean Sea {lat_rng[0]:.2f}–{lat_rng[1]:.2f}°N, "
             f"{abs(lon_rng[1]):.2f}–{abs(lon_rng[0]):.2f}°W "
             f"({int(sst_ocean.sum())} ocean cells)")
print(SST_PLACE)
print(f"period {sst_area.index[0]:%Y-%m-%d} to {sst_area.index[-1]:%Y-%m-%d}, "
      f"{sst_area.notna().sum()} days, mean {sst_area.mean():.2f} °C")

# %% [markdown]
# ## A1. Monthly mean SST for every year (task 1)
# Left: monthly mean SST. Right: the same values minus the climatological mean of each calendar
# month. The anomaly view removes the seasonal cycle, which otherwise dominates the colour
# scale, and makes warm years (e.g. during or after El Niño events) stand out.

# %%
grid = year_month_grid(sst_month)
anom_grid = grid - grid.mean()

fig, axes = plt.subplots(1, 2, figsize=(17, 0.36 * len(grid) + 2.6), layout="constrained")
sns.heatmap(grid, cmap="RdYlBu_r", annot=True, fmt=".1f", annot_kws={"size": 7.5},
            linewidths=0.4, linecolor="white", ax=axes[0],
            cbar_kws={"label": "SST (°C)", "pad": 0.01})
lim = float(np.nanmax(np.abs(anom_grid.values)))
sns.heatmap(anom_grid, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim, annot=True, fmt="+.1f",
            annot_kws={"size": 7.5}, linewidths=0.4, linecolor="white", ax=axes[1],
            cbar_kws={"label": "Anomaly (°C)", "pad": 0.01})
axes[0].set_title("Monthly mean SST")
axes[1].set_title("Monthly SST anomaly (minus the monthly climatology)")
for ax in axes:
    ax.set_xlabel("")
    ax.set_ylabel("Year")
    ax.tick_params(axis="y", rotation=0)
finish(fig, "A1a_monthly_sst_heatmaps.png", SST_PLACE)

# %%
years = grid.index.tolist()
colors = sns.color_palette("viridis", len(years))
fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
for c, y in zip(colors, years):
    ax.plot(range(1, 13), grid.loc[y].values, color=c, lw=1.3, alpha=0.85, marker="o", ms=3)
ax.plot(range(1, 13), grid.mean().values, color="black", lw=3, marker="o", ms=6,
        label="Mean of all years", zorder=5)
sm = plt.cm.ScalarMappable(cmap=ListedColormap(colors),
                           norm=BoundaryNorm(np.arange(len(years) + 1) - 0.5, len(years)))
cb = fig.colorbar(sm, ax=ax, ticks=range(0, len(years), 2), pad=0.01)
cb.ax.set_yticklabels([str(years[i]) for i in range(0, len(years), 2)])
cb.set_label("Year")
ax.set_xticks(range(1, 13), MONTHS)
ax.set_ylabel("Monthly mean SST (°C)")
ax.legend(loc="upper left")
ax.set_title("Monthly mean SST, one line per year")
finish(fig, "A1b_monthly_sst_lines.png", SST_PLACE)

# %% [markdown]
# The annual means put the year-to-year differences in context. The trend is estimated with
# Sen's slope and tested with Mann–Kendall (rank-based, robust to single extreme years). A
# warming trend matters directly for the DHW calculation: the MMM baseline is computed from
# these same years, so any warming inside the record raises the threshold (see A4).

# %%
sst_ann = annual_mean(sst_area)
tr = sen_mk(sst_ann)
fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
ax.plot(tr["x"], tr["y"], color="#888888", lw=1)
ax.scatter(tr["x"], tr["y"], c=[POS if v > tr["y"].mean() else NEG for v in tr["y"]],
           s=70, edgecolor="white", zorder=3)
xx = np.array([tr["x"].min(), tr["x"].max()])
ax.plot(xx, tr["sen_int"] + tr["sen"] * xx, color=POS, lw=2.5,
        label=f"Sen's slope {tr['sen'] * 10:+.2f} °C/decade")
ax.set_xticks(tr["x"][::2])
ax.set_ylabel("Annual mean SST (°C)")
ax.legend(loc="upper left")
text_box(ax, f"95 % CI {tr['sen_lo'] * 10:+.2f} to {tr['sen_hi'] * 10:+.2f} °C/decade\n"
             f"Mann–Kendall τ = {tr['tau']:+.2f}, p = {tr['mk_p']:.3f}")
ax.set_title("Annual mean SST and its trend")
finish(fig, "A1c_annual_sst_trend.png", SST_PLACE)

# %% [markdown]
# ## A2. Climatological monthly cycle (task 2)
# Mean of each calendar month over all years, with the ±1 standard deviation band
# (typical interannual spread) and the min–max envelope (observed extremes). The warmest
# climatological month defines the **Maximum Monthly Mean (MMM)** used for the DHW.
#
# Physical background: the Caribbean SST minimum (Feb–Mar) follows the strongest trade winds
# and the weakest insolation of boreal winter (more evaporation and mixing); the maximum
# (Sep–Oct) comes after the summer insolation peak because the mixed layer stores heat and lags
# the sun by 1–2 months. A dip in July, between two warm months, is the oceanic signature of
# the **Caribbean Low-Level Jet**, which peaks in July and cools the surface by wind mixing,
# evaporation and coastal upwelling (the same mechanism as the "mid-summer drought").

# %%
clim_df = sst_month.to_frame("sst").assign(month=sst_month.index.month).groupby("month")["sst"]
clim = pd.DataFrame({"mean": clim_df.mean(), "std": clim_df.std(), "min": clim_df.min(),
                     "max": clim_df.max(), "n_years": clim_df.count()})
x = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
ax.fill_between(x, clim["min"], clim["max"], color=LINE, alpha=0.14, label="Min–max across years")
ax.fill_between(x, clim["mean"] - clim["std"], clim["mean"] + clim["std"], color=LINE,
                alpha=0.32, label="±1 standard deviation")
ax.plot(x, clim["mean"], color=DARK, lw=3, marker="o", ms=7, label="Climatological mean")
wm, cm_ = int(clim["mean"].idxmax()), int(clim["mean"].idxmin())
ax.annotate(f"MMM (area mean) = {clim['mean'][wm]:.2f} °C", (wm, clim["mean"][wm]),
            xytext=(0, 14), textcoords="offset points", ha="center", color=POS, weight="bold")
ax.annotate(f"minimum {clim['mean'][cm_]:.2f} °C", (cm_, clim["mean"][cm_]),
            xytext=(0, -22), textcoords="offset points", ha="center", color=NEG, weight="bold")
m = clim["mean"]
if m[7] < m[6] and m[7] < m[8]:
    ax.annotate("July dip\n(Caribbean Low-Level Jet)", (7, m[7]), xytext=(0, -46),
                textcoords="offset points", ha="center", fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#777777"))
ax.set_xticks(x, MONTHS)
ax.set_ylabel("SST (°C)")
ax.legend(loc="upper left", frameon=False)
ax.set_title(f"Climatological monthly cycle of SST, {sst_month.index[0]:%Y}–"
             f"{sst_month.index[-1]:%Y}; seasonal range {m.max() - m.min():.2f} °C")
finish(fig, "A2_sst_climatology.png", SST_PLACE)
clim.index = MONTHS
clim.round(2)

# %% [markdown]
# ## A3. Empirical probability and cumulative distributions (task 3)
# The histogram uses the Freedman–Diaconis bin width (adapts to sample size and spread); the
# kernel density estimate is a smoothed guide, not a replacement. The empirical CDF is the
# raw step function of the sorted data. Two physically motivated additions:
#
# * the distribution is drawn separately for the **first and second half** of the record, so a
#   warming shift is visible as a displacement of the whole curve;
# * the CDF is read at the **MMM** and at **MMM + 1 °C**, i.e. how often the area-mean SST is
#   above the coral-stress reference and above the threshold at which DHW starts accumulating.
#
# Daily SST is strongly autocorrelated and seasonal, so the ~7 700 days are far from
# independent samples, and the distribution is skewed / broad because it mixes the cool and
# warm seasons, not because of random noise.

# %%
xs = sst_area.dropna()
mmm_area = float(clim["mean"].max())
half = xs.index.year < xs.index.year[0] + (xs.index.year[-1] - xs.index.year[0] + 1) // 2
parts = {f"{xs.index[0]:%Y}–{xs[half].index[-1]:%Y}": xs[half].values,
         f"{xs[~half].index[0]:%Y}–{xs.index[-1]:%Y}": xs[~half].values}
iqr = np.subtract(*np.percentile(xs, [75, 25]))
bins = int(np.ceil((xs.max() - xs.min()) / (2 * iqr / np.cbrt(len(xs)))))
grid_x = np.linspace(xs.min() - 0.3, xs.max() + 0.3, 400)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
ax = axes[0]
ax.hist(xs, bins=bins, density=True, color=LINE, alpha=0.45, edgecolor="white",
        label=f"Empirical density (n = {len(xs):,} days)")
ax.plot(grid_x, stats.gaussian_kde(xs)(grid_x), color=DARK, lw=2.5, label="KDE, all years")
for (name, v), c in zip(parts.items(), ["#55a868", ANOM_C]):
    ax.plot(grid_x, stats.gaussian_kde(v)(grid_x), color=c, lw=1.8, ls="--", label=f"KDE {name}")
ax.set_xlabel("Daily mean SST (°C)")
ax.set_ylabel("Probability density (°C$^{-1}$)")
ax.set_title("Probability distribution of daily mean SST")

ax = axes[1]
for (name, v), c, lw in [(("all years", xs.values), DARK, 2.6)] + \
        [(kv, c, 1.6) for kv, c in zip(parts.items(), ["#55a868", ANOM_C])]:
    v = np.sort(v)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=lw,
            ls="-" if lw > 2 else "--", label=f"ECDF {name} (median {np.median(v):.2f} °C)")
ax.set_xlabel("Daily mean SST (°C)")
ax.set_ylabel("Cumulative probability  P(SST ≤ x)")
ax.set_ylim(0, 1.02)
ax.set_title("Empirical cumulative distribution of daily mean SST")

for ax in axes:
    for val, c, lab in [(mmm_area, POS, "MMM"), (mmm_area + HOTSPOT_MIN, "#8c2f33", "MMM + 1 °C")]:
        ax.axvline(val, color=c, lw=1.6, ls=":")
        ax.text(val, 0.5, f" {lab}", color=c, fontsize=9, rotation=90, va="center",
                transform=ax.get_xaxis_transform())
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95)
p_mmm = float((xs > mmm_area).mean())
p_mmm1 = float((xs >= mmm_area + HOTSPOT_MIN).mean())
text_box(axes[1], f"P(SST > MMM) = {p_mmm:.1%}\nP(SST ≥ MMM + 1 °C) = {p_mmm1:.2%}",
         "lower right")
finish(fig, "A3_sst_pdf_cdf.png", SST_PLACE)

sst_stats = pd.Series({"n_days": len(xs), "mean": xs.mean(), "median": xs.median(),
                       "std": xs.std(), "min": xs.min(), "max": xs.max(),
                       "p05": xs.quantile(0.05), "p95": xs.quantile(0.95),
                       "skewness": xs.skew(), "excess_kurtosis": xs.kurt(),
                       "shift_of_median_between_halves": np.median(list(parts.values())[1])
                       - np.median(list(parts.values())[0])})
sst_stats.round(3).to_frame("daily area-mean SST")

# %% [markdown]
# ## A4. Degree Heating Weeks (task 4)
#
# **Definition (NOAA Coral Reef Watch).** For every location:
#
# 1. **MMM** – Maximum Monthly Mean: the warmest month of the climatological monthly cycle at
#    that location.
# 2. **HotSpot** – $HS(t) = \max\big(SST_{daily}(t) - MMM,\ 0\big)$.
# 3. Only HotSpots **≥ 1 °C** accumulate: anomalies below 1 °C are assumed insufficient to
#    cause visible stress to corals.
# 4. **DHW** – the sum of qualifying HotSpots over the preceding **12 weeks (84 days)**,
#    divided by 7 to express it in °C-weeks:
#    $DHW(t) = \frac{1}{7}\sum_{i=t-83}^{t} HS_i\,[HS_i \ge 1]$.
#    Two DHW is equivalent to two weeks of constant SST 1 °C above the MMM, or one week at 2 °C.
# 5. **DHW > 4 °C-weeks**: significant bleaching likely (Alert Level 1).
#    **DHW > 8 °C-weeks**: widespread bleaching and some mortality (Alert Level 2).
#
# **What was corrected with respect to the previous version**
#
# * **DHW is computed cell by cell.** Previously the default was the DHW of the domain-mean SST.
#   Averaging first smooths away the warmest cells and blends MMMs of different places; since
#   only HotSpots ≥ 1 °C count, this can turn real local stress into exactly zero. The
#   area-mean version is kept only for comparison in the summary table.
# * **Incomplete windows are not reported.** The first 83 days of the record have less than
#   12 weeks of history, so their DHW is undefined (previously `min_periods=1` reported an
#   underestimate). The rolling sum runs on the continuous series, so January DHW still carries
#   the stress accumulated in the previous autumn.
# * **Regional view.** Besides a single site, the figures show the fraction of the ocean area
#   above each alert level, which is what describes a regional bleaching event.

# %%
def mmm_field(daily, baseline=None):
    """Monthly climatology, MMM and month of the MMM at every grid cell."""
    monthly = daily.resample(time="MS").mean()
    if baseline is not None:
        monthly = monthly.sel(time=slice(f"{baseline[0]}-01-01", f"{baseline[1]}-12-31"))
    clim = monthly.groupby("time.month").mean("time")
    mmm = clim.max("month")
    month = (clim.fillna(-np.inf).argmax("month") + 1).where(mmm.notnull())
    return clim, mmm, month


def degree_heating_weeks(daily, mmm, threshold=HOTSPOT_MIN):
    """HotSpot and DHW (degC-weeks) following NOAA Coral Reef Watch.
    `threshold` is the minimum HotSpot that accumulates (1 degC in CRW)."""
    hotspot = (daily - mmm).clip(min=0)                      # NaN stays NaN
    qualifying = hotspot.where(hotspot >= threshold, 0.0).where(daily.notnull())
    dhw = qualifying.rolling(time=DHW_WINDOW_DAYS, min_periods=DHW_WINDOW_DAYS).sum() / 7.0
    return hotspot, dhw


available = sorted(set(sst.time.dt.year.values))
DHW_YEARS = tuple(y for y in DHW_YEARS if y in available)
assert len(DHW_YEARS) >= 1, f"DHW_YEARS must be in the record {available}"

sst_clim, mmm, mmm_month = mmm_field(sst, MMM_BASELINE)
hotspot, dhw = degree_heating_weeks(sst, mmm)
peak = dhw.groupby("time.year").max("time")                  # annual peak DHW per cell
base_txt = (f"{MMM_BASELINE[0]}–{MMM_BASELINE[1]}" if MMM_BASELINE
            else f"{available[0]}–{available[-1]}")

# Site for the detailed time series: a given reef, or the most stressed cell.
if SST_SITE is not None:
    site = nearest_ocean_cell(sst, *SST_SITE)
    site_txt = f"site {fmt_coord(site['lat'], site['lon'])} (nearest ocean cell to {fmt_coord(*SST_SITE)})"
else:
    # combined stress of the selected years, so the cell is relevant to both of them
    pk = peak.sel(year=list(DHW_YEARS)).sum("year", min_count=1)
    ij = pk.fillna(-1).argmax(dim=["latitude", "longitude"])
    site = {"lat": float(sst.latitude[ij["latitude"]]), "lon": float(sst.longitude[ij["longitude"]])}
    site_txt = f"most stressed cell in {', '.join(map(str, DHW_YEARS))}: {fmt_coord(site['lat'], site['lon'])}"
print(f"MMM across the domain: {float(mmm.min()):.2f}–{float(mmm.max()):.2f} °C "
      f"(baseline {base_txt})")
print("Detail series:", site_txt)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), layout="constrained")
ax = axes[0]
mesh = ax.pcolormesh(mmm.longitude, mmm.latitude, mmm, cmap="inferno", shading="nearest")
fig.colorbar(mesh, ax=ax, label="MMM (°C)", fraction=0.035, pad=0.02)
ax.set_title(f"Maximum Monthly Mean (baseline {base_txt})")
ax = axes[1]
present = sorted(set(int(v) for v in np.unique(mmm_month.values[np.isfinite(mmm_month.values)])))
cm_months = ListedColormap(sns.color_palette("Spectral_r", len(present)))
idx = xr.apply_ufunc(lambda v: np.searchsorted(present, v), mmm_month.fillna(present[0])).where(mmm_month.notnull())
mesh = ax.pcolormesh(mmm.longitude, mmm.latitude, idx, cmap=cm_months, shading="nearest",
                     norm=BoundaryNorm(np.arange(len(present) + 1) - 0.5, len(present)))
cb = fig.colorbar(mesh, ax=ax, ticks=range(len(present)), fraction=0.035, pad=0.02)
cb.ax.set_yticklabels([MONTHS[k - 1] for k in present])
ax.set_title("Month of the MMM")
for ax in axes:
    map_axes(ax, mmm)
    ax.plot(site["lon"], site["lat"], marker="*", ms=15, color="#00b4d8", mec="black", ls="none")
finish(fig, "A4a_mmm_maps.png", SST_PLACE)

# %% [markdown]
# **Peak DHW maps.** Highest DHW reached in each selected year at every grid cell, with the
# 4 and 8 °C-week contours. Colour classes follow the alert levels.

# %%
levels = [0, 1, 2, 4, 6, 8, 12, 16, 20]
cmap_dhw = ListedColormap(["#f7f7f7", "#fff3b0", "#ffd166", "#f4a261", "#e76f51",
                           "#c1121f", "#780000", "#3d0000"]).with_extremes(over="#1a0000")
norm_dhw = BoundaryNorm(levels, cmap_dhw.N)
fig, axes = plt.subplots(1, len(DHW_YEARS), figsize=(8 * len(DHW_YEARS), 5.2),
                         layout="constrained", squeeze=False)
for ax, y in zip(axes[0], DHW_YEARS):
    pm = peak.sel(year=y)
    mesh = ax.pcolormesh(pm.longitude, pm.latitude, pm, cmap=cmap_dhw, norm=norm_dhw, shading="nearest")
    if float(pm.max()) >= ALERT1:
        cs = ax.contour(pm.longitude, pm.latitude, pm, levels=[l for l in (ALERT1, ALERT2) if l <= float(pm.max())],
                        colors=["#333333", "black"], linewidths=[1.2, 2.0])
        ax.clabel(cs, fmt="%g", fontsize=8)
    a4 = float(area_fraction(pm >= ALERT1, sst_ocean))
    a8 = float(area_fraction(pm >= ALERT2, sst_ocean))
    ax.plot(site["lon"], site["lat"], marker="*", ms=15, color="#00b4d8", mec="black", ls="none")
    map_axes(ax, pm)
    ax.set_title(f"{y}: max {float(pm.max()):.1f} °C-weeks | area ≥ 4: {a4:.0%}, ≥ 8: {a8:.0%}")
fig.colorbar(mesh, ax=axes[0].tolist(), label="Annual peak DHW (°C-weeks)", ticks=levels,
             extend="max", shrink=0.8, pad=0.01)
finish(fig, "A4b_peak_dhw_maps.png", f"{SST_PLACE}; MMM baseline {base_txt}; star = detail site")

# %% [markdown]
# **Regional evolution through the year.** Top: the highest DHW anywhere in the domain on each
# day. Bottom: the share of the ocean area at or above each alert level. Both years are drawn on
# a common seasonal axis, which shows when stress began, how fast it built up and how long it
# lasted, not only how high it peaked.

# %%
dhw_max = dhw.max(("latitude", "longitude")).to_series()
frac4 = area_fraction(dhw >= ALERT1, sst_ocean).to_series().where(dhw_max.notna())
frac8 = area_fraction(dhw >= ALERT2, sst_ocean).to_series().where(dhw_max.notna())
pal = sns.color_palette("Set1", max(3, len(DHW_YEARS)))
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, layout="constrained")
top = max(ALERT2 * 1.25, float(dhw_max[dhw_max.index.year.isin(DHW_YEARS)].max()) * 1.15)
for lo, hi, c in [(0, ALERT1, "#f2f7fb"), (ALERT1, ALERT2, "#fde8d4"), (ALERT2, top, "#f7cfcf")]:
    axes[0].axhspan(lo, hi, color=c, zorder=0)
for lvl, lab in [(ALERT1, "Alert 1: bleaching likely"), (ALERT2, "Alert 2: widespread bleaching, mortality")]:
    axes[0].axhline(lvl, color="#999999", ls="--", lw=1)
    axes[0].text(pd.Timestamp(REF_YEAR, 1, 5), lvl, f" {lab}", va="bottom", fontsize=9, color="#555555")
for c, y in zip(pal, DHW_YEARS):
    s = dhw_max[dhw_max.index.year == y]
    axes[0].plot(seasonal_axis(s.index), s.values, color=c, lw=2.6,
                 label=f"{y} (peak {s.max():.1f} °C-weeks on {s.idxmax():%d %b})")
    for f_, ls, lab in [(frac4, "-", "≥ 4"), (frac8, "--", "≥ 8")]:
        s = f_[f_.index.year == y] * 100
        axes[1].plot(seasonal_axis(s.index), s.values, color=c, lw=2.2, ls=ls,
                     label=f"{y}: DHW {lab} °C-weeks (max {s.max():.0f} %)")
axes[0].set_ylim(0, top)
axes[0].set_ylabel("Domain-maximum DHW (°C-weeks)")
axes[0].legend(loc="upper left")
axes[1].set_ylabel("Ocean area (%)")
axes[1].set_ylim(0, 105)
axes[1].legend(loc="upper left", ncol=2, fontsize=9)
month_axis(axes[1])
axes[0].set_title("Accumulated thermal stress across the domain")
finish(fig, "A4c_dhw_regional_evolution.png", f"{SST_PLACE}; MMM baseline {base_txt}")

# %% [markdown]
# **Site detail: thermal stress and DHW, year by year.** Each panel overlays the daily SST
# (left axis) on the DHW it produces (right axis). The SST is read against the MMM (dashed) and
# the MMM + 1 °C threshold (dotted); only the red-shaded excess above that threshold accumulates.
# The area under the DHW curve is coloured by the alert level reached. A year whose SST sat
# 0.9 °C above the MMM for three months still scores DHW = 0, exactly like a year that never
# reached the MMM – the SST trace makes that difference visible.

# %%
s_sst, s_mmm = cell_series(sst, site), float(mmm.sel(latitude=site["lat"], longitude=site["lon"]))
s_dhw = cell_series(dhw, site)
s_hs = (s_sst - s_mmm).clip(lower=0)

DHW_C = "#b5495b"                                            # DHW line, as in A4c/A4e
SST_C, THR_C = "#1d2b3a", "#8c2f33"
LEVEL_FILLS = [(0, ALERT1, "#fde8c4", f"DHW < {ALERT1:g}: no alert (stress watch)"),
               (ALERT1, ALERT2, "#f4a261", f"Alert Level 1 ({ALERT1:g}–{ALERT2:g} °C-weeks): bleaching likely"),
               (ALERT2, np.inf, "#9b2226", f"Alert Level 2 (≥ {ALERT2:g} °C-weeks): widespread bleaching, mortality")]


def alert_name(pk):
    return "Alert Level 2" if pk >= ALERT2 else "Alert Level 1" if pk >= ALERT1 else "no alert"


def dhw_axis_limits(sst_s, dhw_list, years, thr_max=HOTSPOT_MIN):
    """Shared limits: DHW fills the lower ~45 % of a panel, SST the upper part."""
    sel = sst_s.index.year.isin(years)
    peak_all = max(float(np.nanmax(d[sel])) for d in dhw_list)
    dhw_top = max(ALERT2 * 1.25, peak_all * 1.15) / 0.45
    lo = float(sst_s[sel].min())
    hi = float(max(sst_s[sel].max(), s_mmm + thr_max))
    return dhw_top, (lo, hi)


def dhw_panel(ax, year, sst_s, dhw_s, mmm_v, thr, dhw_top, sst_lims,
              left_label=True, right_label=True):
    """Daily SST (left axis) against MMM and MMM + thr, and the DHW it produces
    (right axis) with the area under the curve coloured by alert level."""
    sel = sst_s.index.year == year
    t, sy, dy = sst_s.index[sel], sst_s[sel], dhw_s[sel].fillna(0)
    hy = (sy - mmm_v).clip(lower=0)
    counts = (hy >= thr) & (hy > 0)                          # days that add to the DHW

    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)                      # SST drawn over the fills
    ax.patch.set_visible(False)
    for lo, hi, c, _ in LEVEL_FILLS:
        ax2.fill_between(t, lo, np.minimum(dy, hi), where=dy > lo, color=c,
                         alpha=0.85, lw=0, interpolate=True)
    ax2.plot(t, dy, color=DHW_C, lw=2.2)
    for lvl, ls in [(ALERT1, (0, (2, 2))), (ALERT2, (0, (1, 1.5)))]:
        ax2.axhline(lvl, color=DHW_C, lw=1.1, ls=ls, alpha=0.8)
    ax2.set_ylim(0, dhw_top)
    ax2.set_yticks([v for v in (0, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40) if v <= dhw_top * 0.5])
    ax2.tick_params(axis="y", colors=DHW_C)
    ax2.grid(False)
    if right_label:
        ax2.set_ylabel("DHW (°C-weeks)", color=DHW_C)
        ax2.yaxis.set_label_coords(1.05, 0.22)
    else:
        ax2.set_yticklabels([])

    lo, hi = sst_lims
    span = hi - lo
    ax.plot(t, sy, color=SST_C, lw=1.3)
    ax.axhline(mmm_v, color=NEG, lw=1.5, ls="--")
    if thr > 0:
        ax.axhline(mmm_v + thr, color=THR_C, lw=1.5, ls=":")
    ax.fill_between(t, mmm_v + thr, sy, where=counts, color=POS, alpha=0.45, lw=0,
                    interpolate=True)
    ax.set_ylim(lo - 0.95 * span, hi + 0.12 * span)
    # SST ticks (and their grid lines) only where SST is drawn, not over the DHW area
    ax.set_yticks([v for v in ax.get_yticks() if lo - 0.25 <= v <= hi + 0.12 * span])
    if left_label:
        ax.set_ylabel("SST (°C)")
        ax.yaxis.set_label_coords(-0.05, 0.72)
    ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="x", color="#e6e6e6")

    pk = float(dy.max())
    if pk > 0:
        ax2.annotate(f"{pk:.1f} °C-weeks\n{dy.idxmax():%d %b}", (dy.idxmax(), pk),
                     xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, color=DHW_C, weight="bold")
    return {"peak": pk, "date": dy.idxmax() if pk > 0 else None, "level": alert_name(pk),
            "days_above_mmm": int((hy > 0).sum()), "days_counted": int(counts.sum())}


def dhw_legend(fig, thr_label):
    handles = [Line2D([], [], color=SST_C, lw=1.6, label="Daily mean SST"),
               Line2D([], [], color=NEG, lw=1.5, ls="--", label=f"MMM = {s_mmm:.2f} °C"),
               Line2D([], [], color=THR_C, lw=1.5, ls=":", label=thr_label),
               Patch(color=POS, alpha=0.45, label="HotSpot that adds to the DHW"),
               Line2D([], [], color=DHW_C, lw=2.2, label="DHW (right axis)")]
    handles += [Patch(color=c, alpha=0.85, label=lab) for *_, c, lab in LEVEL_FILLS]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)


dhw_top, sst_lims = dhw_axis_limits(s_sst, [s_dhw], DHW_YEARS)
fig, axes = plt.subplots(len(DHW_YEARS), 1, figsize=(14, 4.6 * len(DHW_YEARS)),
                         layout="constrained", squeeze=False)
for ax, y in zip(axes[:, 0], DHW_YEARS):
    r = dhw_panel(ax, y, s_sst, s_dhw, s_mmm, HOTSPOT_MIN, dhw_top, sst_lims)
    ax.set_title(f"{y}  ·  peak DHW {r['peak']:.1f} °C-weeks ({r['level']})  ·  "
                 f"{r['days_above_mmm']} days above MMM, {r['days_counted']} with HotSpot ≥ 1 °C",
                 loc="left", fontsize=11.5)
fig.suptitle(f"Thermal stress and Degree Heating Weeks – ERA5 SST, {site_txt}\n"
             f"MMM baseline {base_txt}; DHW = 12-week sum of HotSpots ≥ {HOTSPOT_MIN:g} °C / 7",
             weight="bold")
dhw_legend(fig, f"MMM + {HOTSPOT_MIN:g} °C (accumulation threshold)")
finish(fig, "A4d_dhw_site_detail.png")

# %% [markdown]
# **Sensitivity of the DHW to the HotSpot threshold.** The NOAA CRW definition only accumulates
# HotSpots ≥ 1 °C, on the assumption that smaller anomalies do not cause visible stress. To see
# how much that choice controls the result, the DHW is recomputed with three thresholds:
#
# | Scenario | HotSpots that accumulate | Role |
# |---|---|---|
# | ≥ 1 °C | SST ≥ MMM + 1 °C | NOAA CRW definition (the one used everywhere else in A4) |
# | ≥ 0.5 °C | SST ≥ MMM + 0.5 °C | intermediate sensitivity test |
# | > 0 | any SST above the MMM | upper bound: every warm-season excess counts |
#
# Rows are scenarios, columns are years; all panels share the same SST and DHW axes, so the
# heights are directly comparable. Lowering the threshold can only add days, so the DHW grows
# from top to bottom. Only the ≥ 1 °C row is comparable with the 4 / 8 °C-week alert levels,
# which were calibrated against observed bleaching with that definition; the other two rows show
# how sensitive the alert assessment is to the threshold, not alternative alert levels.

# %%
DHW_SCENARIOS = {"HotSpot ≥ 1 °C (NOAA CRW)": 1.0,
                 "HotSpot ≥ 0.5 °C": 0.5,
                 "HotSpot > 0 (any SST above the MMM)": 0.0}

site_sst_da = sst.sel(latitude=site["lat"], longitude=site["lon"])
site_mmm_da = mmm.sel(latitude=site["lat"], longitude=site["lon"])
scen_site, scen_field = {}, {}
for name, thr in DHW_SCENARIOS.items():
    scen_site[name] = degree_heating_weeks(site_sst_da, site_mmm_da, thr)[1].to_series()
    scen_field[name] = degree_heating_weeks(sst, mmm, thr)[1]

dhw_top_s, sst_lims_s = dhw_axis_limits(s_sst, list(scen_site.values()), DHW_YEARS)
fig, axes = plt.subplots(len(DHW_SCENARIOS), len(DHW_YEARS),
                         figsize=(9 * len(DHW_YEARS), 4.1 * len(DHW_SCENARIOS)),
                         layout="constrained", squeeze=False, sharex="col")
rows = []
for i, (name, thr) in enumerate(DHW_SCENARIOS.items()):
    peak_f = scen_field[name].groupby("time.year").max("time")
    for j, y in enumerate(DHW_YEARS):
        ax = axes[i, j]
        r = dhw_panel(ax, y, s_sst, scen_site[name], s_mmm, thr, dhw_top_s, sst_lims_s,
                      left_label=(j == 0), right_label=(j == len(DHW_YEARS) - 1))
        ax.set_title(f"{name}  ·  {y}\npeak {r['peak']:.1f} °C-weeks ({r['level']}), "
                     f"{r['days_counted']} days accumulate", loc="left", fontsize=10.5)
        pm = peak_f.sel(year=y)
        rows.append({"scenario": name, "year": y, "threshold (°C)": thr,
                     "site peak DHW": r["peak"],
                     "site peak date": r["date"].date() if r["date"] is not None else None,
                     "site alert": r["level"], "site days accumulating": r["days_counted"],
                     "domain peak DHW": float(pm.max()),
                     "% area DHW ≥ 4": float(area_fraction(pm >= ALERT1, sst_ocean)) * 100,
                     "% area DHW ≥ 8": float(area_fraction(pm >= ALERT2, sst_ocean)) * 100})
fig.suptitle(f"DHW under three HotSpot thresholds – ERA5 SST, {site_txt}\n"
             f"MMM baseline {base_txt}; 12-week accumulation; shared axes in every panel",
             weight="bold")
dhw_legend(fig, "MMM + threshold (not drawn when the threshold is 0)")
finish(fig, "A4f_dhw_threshold_scenarios.png")

scen_table = pd.DataFrame(rows).set_index(["scenario", "year"])
scen_table.round(2).to_csv(FIG_DIR / "A4f_dhw_threshold_scenarios.csv")
for y in DHW_YEARS:
    sub = scen_table.xs(y, level="year")
    p = sub["site peak DHW"]
    print(f"{y}: site peak DHW " + ", ".join(f"{v:.1f}" for v in p) + " °C-weeks for thresholds "
          + ", ".join(f"{t:g}" for t in sub["threshold (°C)"]) + " °C; "
          f"area ≥ 4: " + ", ".join(f"{v:.0f} %" for v in sub["% area DHW ≥ 4"])
          + "; area ≥ 8: " + ", ".join(f"{v:.0f} %" for v in sub["% area DHW ≥ 8"]))
scen_table.round(2)

# %% [markdown]
# **Record context.** Two years in isolation cannot say whether a value is unusual. Here every
# year of the record is shown, with the selected years highlighted.

# %%
area4 = area_fraction(peak >= ALERT1, sst_ocean).to_series() * 100
area8 = area_fraction(peak >= ALERT2, sst_ocean).to_series() * 100
dom_peak = peak.max(("latitude", "longitude")).to_series()
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout="constrained")
hl = [y in DHW_YEARS for y in dom_peak.index]
axes[0].bar(dom_peak.index, dom_peak, color=["#b5495b" if h else "#c3cfe0" for h in hl])
for lvl in (ALERT1, ALERT2):
    axes[0].axhline(lvl, color="#999999", ls="--", lw=1)
axes[0].set_ylabel("Domain peak DHW\n(°C-weeks)")
axes[1].bar(area4.index, area4, color=["#f4a261" if h else "#e3d5c3" for h in hl], label="DHW ≥ 4")
axes[1].bar(area8.index, area8, color=["#780000" if h else "#9c8c7c" for h in hl], label="DHW ≥ 8")
axes[1].set_ylabel("Ocean area reaching\nthe level in the year (%)")
axes[1].legend(loc="upper left")
axes[1].set_xticks(dom_peak.index)
axes[1].tick_params(axis="x", rotation=45)
axes[0].set_title(f"Annual thermal stress over the whole record (highlighted: "
                  f"{', '.join(map(str, DHW_YEARS))}); {available[0]} starts after day 84")
finish(fig, "A4e_dhw_record_context.png", f"{SST_PLACE}; MMM baseline {base_txt}")

# %%
# Literal "regional" variant, for comparison only: DHW of the area-mean SST series.
area_da = xr.DataArray(sst_area.values, coords={"time": sst_area.index}, dims="time")
_, mmm_reg, _ = mmm_field(area_da, MMM_BASELINE)
_, dhw_reg = degree_heating_weeks(area_da, mmm_reg)
dhw_reg = dhw_reg.to_series()

rows = []
for y in DHW_YEARS:
    sel = s_dhw.index.year == y
    rows.append({
        "year": y,
        "domain peak DHW": float(dom_peak[y]),
        "% area DHW ≥ 4": float(area4[y]), "% area DHW ≥ 8": float(area8[y]),
        "site peak DHW": float(s_dhw[sel].max()),
        "site peak date": s_dhw[sel].idxmax().date() if s_dhw[sel].max() > 0 else None,
        "site days HotSpot ≥ 1": int((s_hs[sel] >= HOTSPOT_MIN).sum()),
        "site max SST": float(s_sst[sel].max()),
        "area-mean-series peak DHW": float(dhw_reg[dhw_reg.index.year == y].max()),
    })
dhw_table = pd.DataFrame(rows).set_index("year")
dhw_table.round(2).to_csv(FIG_DIR / "A4_dhw_summary.csv")
for y, r in dhw_table.iterrows():
    lvl = ("Alert Level 2 – widespread bleaching and some mortality expected"
           if r["domain peak DHW"] >= ALERT2 else
           "Alert Level 1 – significant bleaching likely" if r["domain peak DHW"] >= ALERT1
           else "below the bleaching thresholds")
    print(f"{y}: domain peak {r['domain peak DHW']:.1f} °C-weeks ({lvl}); "
          f"{r['% area DHW ≥ 4']:.0f} % of the area ≥ 4, {r['% area DHW ≥ 8']:.0f} % ≥ 8. "
          f"Area-mean series would give only {r['area-mean-series peak DHW']:.1f}.")
dhw_table.round(2)

# %% [markdown]
# ### Interpretation and limitations of the DHW results
#
# * **Reading the index.** DHW combines the *intensity* and the *duration* of heat stress. It
#   rises only on days when SST is ≥ 1 °C above the local MMM and decays 12 weeks later, so the
#   peak typically comes at the end of the warm season (Sep–Nov in the Caribbean) and late-season
#   events can carry stress into the following January.
# * **Why the per-cell result differs from the area-mean series.** Heat waves are patchy; the
#   area mean smooths the hottest cells and the domain-mean MMM is not the MMM any coral
#   experiences. The table quantifies this underestimate.
# * **Baseline.** NOAA CRW uses a 1985–2012 climatology re-centred to 1988.3. Here the MMM is
#   built from the record itself (2000–2020 by default). Because the Caribbean warmed during
#   these years (A1c), this MMM is higher than CRW's, so **DHW values here are conservative**
#   (lower than the official product), and a warming trend is partly absorbed into the
#   baseline. Set `MMM_BASELINE` to test the sensitivity.
# * **Data.** ERA5 SST (~0.25°) is a daily foundation-like analysis, coarser than the 5 km CRW
#   product, and is not restricted to reef pixels; near-shore cells and small reefs are not
#   resolved. NOAA also uses night-time SST to avoid diurnal warming; daily means are close to
#   that for ERA5, which has little diurnal cycle.
# * **Climate context.** The return time between severe bleaching events has shortened from
#   ~25–30 years in the 1980s to ~6 years by 2016 (Hughes et al. 2018). Rising baseline SST
#   means the same interannual variability (ENSO, Atlantic warm pool) crosses the MMM + 1 °C
#   threshold more often.

# %% [markdown]
# ---
# # Part B – Significant wave height at Tumaco (Colombian Pacific)
#
# **Physical setting.** Tumaco lies in the Panama Bight, in the eastern tropical Pacific, where
# local winds are weak. Hs there is a mix of:
# * **long-period swell from the Southern Ocean / South Pacific** (dominant, strongest in the
#   austral winter–spring, Jun–Oct);
# * **local wind sea** driven by the cross-equatorial southerlies and the Chocó jet, which
#   strengthen when the ITCZ moves north (boreal summer–autumn);
# * **North Pacific swell** in boreal winter, which reaches the coast more often during El Niño
#   winters when the Aleutian low and North Pacific storm track intensify.
#
# The series is taken at the grid cell nearest to `HS_SITE` (no spatial smoothing); set it to
# `None` for the area mean.

# %%
hs = open_daily(HS_DAILY_FILE, HS_VAR)
if HS_SITE is not None:
    hs_cell = nearest_ocean_cell(hs, *HS_SITE)
    hs_d = cell_series(hs, hs_cell)
    HS_PLACE = (f"ERA5 Hs, grid cell {fmt_coord(hs_cell['lat'], hs_cell['lon'])} "
                f"({hs_cell['dist_km']:.0f} km from {fmt_coord(*HS_SITE)}), Tumaco")
    if hs_cell["nearest_is_land"]:
        print("NOTE: the nearest cell is land; the nearest ocean cell is used.")
else:
    hs_cell = None
    hs_d = area_mean(hs)
    HS_PLACE = f"ERA5 Hs, area mean of {int(ocean_mask(hs).sum())} ocean cells, Tumaco"
hs_d.name = "hs"
hs_m = monthly_mean(hs_d)
if hs_m.isna().any():
    raise ValueError("Monthly series has gaps; spectra and filters need a continuous record.")
print(HS_PLACE)
print(f"{hs_d.index[0]:%Y-%m-%d} to {hs_d.index[-1]:%Y-%m-%d}: {hs_d.notna().sum()} days, "
      f"{len(hs_m)} months, mean Hs {hs_d.mean():.3f} m")

fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
mean_hs = hs.mean("time")
mesh = ax.pcolormesh(hs.longitude, hs.latitude, mean_hs, cmap="YlGnBu", shading="nearest")
fig.colorbar(mesh, ax=ax, label="Mean Hs (m)", fraction=0.04)
if hs_cell is not None:
    ax.plot(hs_cell["lon"], hs_cell["lat"], marker="s", ms=10, mfc="none", mec=POS, mew=2,
            ls="none", label="Selected cell")
    ax.plot(HS_SITE[1], HS_SITE[0], marker="*", ms=14, color=DARK, mec="white", ls="none",
            label="Requested point")
    ax.legend(loc="lower left", fontsize=9)
map_axes(ax, hs)
ax.set_title("Mean Hs and selected location")
finish(fig, "B0_location.png", HS_PLACE)


# %%
# ENSO reference: Oceanic Nino Index (NOAA CPC). Used in B4 and B7; optional.
SEASONS = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
           "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def load_oni():
    """CPC 'oni.ascii.txt' (season, year, total, anomaly); each 3-month season
    is dated to its centre month. Downloaded once and cached if possible."""
    try:
        if not ONI_FILE.exists():
            req = urllib.request.Request(ONI_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                ONI_FILE.write_text(r.read().decode("utf-8", "replace"))
        rows = [(pd.Timestamp(int(p[1]), SEASONS[p[0]], 1), float(p[3]))
                for p in (ln.split() for ln in ONI_FILE.read_text().splitlines())
                if len(p) >= 4 and p[0] in SEASONS]
        oni = pd.Series(dict(rows)).sort_index().rename("oni")
        print(f"ONI {oni.index[0]:%Y-%m} to {oni.index[-1]:%Y-%m}")
        return oni
    except Exception as exc:                                      # noqa: BLE001
        print(f"ONI not available ({exc}); ENSO overlays skipped. "
              f"Download {ONI_URL} next to the notebook to enable them.")
        return None


def enso_phases(oni, threshold=0.5, min_len=5):
    """CPC episodes: El Nino (+1) / La Nina (-1) when |ONI| >= 0.5 degC for at
    least 5 consecutive overlapping seasons."""
    phase = pd.Series(0, index=oni.index, dtype=int)
    for sign in (1, -1):
        hit = (sign * oni >= threshold).astype(int)
        run_id = (hit.diff() != 0).cumsum()
        run_len = hit.groupby(run_id).transform("sum")
        phase[(hit == 1) & (run_len >= min_len)] = sign
    return phase


def shade_enso(ax, phase):
    for sign, color, name in [(1, NINO_BG, "El Niño"), (-1, NINA_BG, "La Niña")]:
        on = (phase == sign).astype(int)
        starts = phase.index[(on.diff().fillna(on.iloc[0]) == 1)]
        ends = phase.index[(on.diff().shift(-1).fillna(-on.iloc[-1]) == -1)]
        for i, (a, b) in enumerate(zip(starts, ends)):
            ax.axvspan(a, b + pd.offsets.MonthBegin(1), color=color, lw=0, zorder=0,
                       label=name if i == 0 else None)


oni = load_oni()
phase = enso_phases(oni).reindex(hs_m.index) if oni is not None else None

# %% [markdown]
# ## B1. Daily mean Hs for every year (task 11)
# Heat map: every day of the record (rows = years). The colour scale spans the 0.5–99.5th
# percentiles so ordinary days are not washed out by a few storms. Small multiples: each year
# against the all-years mean and 10–90 % band, on a shared scale.

# %%
d = hs_d.to_frame("hs").assign(year=hs_d.index.year, doy=doy365(hs_d.index))
daily_grid = d.groupby(["year", "doy"])["hs"].mean().unstack().reindex(columns=range(1, 366))
lo, hi = np.nanpercentile(daily_grid.values, [0.5, 99.5])
fig, ax = plt.subplots(figsize=(15, 0.32 * len(daily_grid) + 2.4), layout="constrained")
mesh = ax.pcolormesh(np.arange(0.5, 366), np.arange(len(daily_grid) + 1), daily_grid.values,
                     cmap="YlGnBu", vmin=lo, vmax=hi)
ax.set_yticks(np.arange(len(daily_grid)) + 0.5, daily_grid.index.astype(str), fontsize=9)
ax.invert_yaxis()
ax.set_xticks([pd.Timestamp(REF_YEAR, m, 15).dayofyear for m in range(1, 13)], MONTHS)
ax.grid(False)
fig.colorbar(mesh, ax=ax, extend="both", pad=0.01, label="Daily mean Hs (m)")
ax.set_title("Daily mean significant wave height, year by year")
finish(fig, "B1a_daily_hs_heatmap.png", HS_PLACE)

yrs = daily_grid.index.tolist()
xd = pd.date_range(f"{REF_YEAR}-01-01", periods=365, freq="D")
ncol = 3
nrow = int(np.ceil(len(yrs) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.1 * nrow + 1), sharex=True, sharey=True,
                         layout="constrained", squeeze=False)
q10, q90, qm = daily_grid.quantile(0.1), daily_grid.quantile(0.9), daily_grid.mean()
for ax, y in zip(axes.flat, yrs):
    ax.fill_between(xd, q10, q90, color="#dddddd", lw=0)
    ax.plot(xd, qm, color="#8a8a8a", lw=1)
    ax.plot(xd, daily_grid.loc[y], color=LINE, lw=1)
    ax.set_title(f"{y}   mean {np.nanmean(daily_grid.loc[y]):.2f} m", fontsize=10, loc="left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
for ax in axes.flat[len(yrs):]:
    ax.set_visible(False)
fig.supylabel("Daily mean Hs (m)")
fig.legend(handles=[Line2D([], [], color=LINE, label="Daily mean Hs"),
                    Line2D([], [], color="#8a8a8a", label="Mean of all years"),
                    Patch(color="#dddddd", label="10th–90th percentile of all years")],
           loc="outside upper center", ncol=3, frameon=False)
finish(fig, "B1b_daily_hs_by_year.png", HS_PLACE)

# %% [markdown]
# ## B2. Monthly mean Hs for every year (task 12)

# %%
hs_grid = year_month_grid(hs_m)
fig, axes = plt.subplots(1, 2, figsize=(18, 0.34 * len(hs_grid) + 2.6), layout="constrained",
                         gridspec_kw={"width_ratios": [1.15, 1]})
sns.heatmap(hs_grid, cmap="YlGnBu", annot=True, fmt=".2f", annot_kws={"size": 7.5},
            linewidths=0.4, linecolor="white", ax=axes[0],
            cbar_kws={"label": "Monthly mean Hs (m)", "pad": 0.01})
axes[0].set_xlabel("")
axes[0].set_ylabel("Year")
axes[0].tick_params(axis="y", rotation=0)
axes[0].set_title("Monthly mean Hs")
ax = axes[1]
cols = sns.color_palette("viridis", len(hs_grid))
for c, y in zip(cols, hs_grid.index):
    ax.plot(range(1, 13), hs_grid.loc[y], color=c, lw=1.2, alpha=0.8, marker="o", ms=2.5)
ax.plot(range(1, 13), hs_grid.mean(), color="black", lw=3, marker="o", ms=6, label="Mean of all years")
sm = plt.cm.ScalarMappable(cmap=ListedColormap(cols),
                           norm=BoundaryNorm(np.arange(len(hs_grid) + 1) - 0.5, len(hs_grid)))
cb = fig.colorbar(sm, ax=ax, ticks=range(0, len(hs_grid), 2), pad=0.01)
cb.ax.set_yticklabels([str(hs_grid.index[i]) for i in range(0, len(hs_grid), 2)])
ax.set_xticks(range(1, 13), MONTHS)
ax.set_ylabel("Monthly mean Hs (m)")
ax.legend(loc="upper left")
ax.set_title("One line per year")
finish(fig, "B2_monthly_hs.png", HS_PLACE)

# %% [markdown]
# ## B3. Annual increase (trend) of Hs (task 13)
#
# The trend is estimated three ways:
# * **daily** and **monthly** means: least squares with 3 annual harmonics, so the seasonal
#   cycle is modelled rather than left in the residuals (otherwise the trend depends on which
#   season the record starts and ends in). Confidence intervals use the effective sample size
#   of the autocorrelated residuals; treating 7 700 correlated days as independent would make
#   almost any slope look "significant".
# * **annual** means and **annual 95th percentiles** of daily Hs: Sen's slope with
#   Mann–Kendall test (robust, ~independent annual values). Climate-change signals in waves are
#   often clearer in the extremes than in the mean, hence the P95.

# %%
res_d = harmonic_trend(hs_d, "daily")
res_m = harmonic_trend(hs_m, "monthly")
ann_mean = annual_mean(hs_d)
ann_p95 = hs_d.groupby(hs_d.index.year).quantile(0.95)[ann_mean.index]
tr_mean, tr_p95 = sen_mk(ann_mean), sen_mk(ann_p95)


def trend_text(r):
    b, (lo, hi) = r["slope"], r["ci"]
    return (f"trend {b * 100:+.2f} cm/yr ({b * 1000:+.1f} cm/decade, "
            f"{b * 10 / r['mean'] * 100:+.1f} % per decade)\n"
            f"95 % CI {lo * 100:+.2f} to {hi * 100:+.2f} cm/yr, p = {r['p']:.3f}\n"
            f"r1 = {r['r1']:.2f}, N_eff = {r['n_eff']:,.0f} of {r['n']:,}")


fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, layout="constrained")
ax = axes[0]
ax.plot(hs_d.index, hs_d, color="#9ecae1", lw=0.5, label="Daily mean Hs")
ax.plot(hs_d.index, hs_d.rolling(365, center=True, min_periods=330).mean(), color=DARK, lw=2.2,
        label="365-day running mean")
ax.plot(hs_d.index, res_d["fit"], color=POS, lw=2.5, ls="--", label="Linear trend")
text_box(ax, trend_text(res_d))
ax.legend(loc="upper left", fontsize=9)
ax.set_ylabel("Hs (m)")
ax.set_title("Daily means")
ax = axes[1]
xm = mid_month(hs_m.index)
ax.plot(xm, hs_m, color=LINE, lw=1, marker="o", ms=2.5, label="Monthly mean Hs")
ax.plot(xm, hs_m.rolling(12, center=True, min_periods=10).mean(), color=DARK, lw=2.2,
        label="12-month running mean")
ax.plot(xm, res_m["fit"], color=POS, lw=2.5, ls="--", label="Linear trend")
text_box(ax, trend_text(res_m))
ax.legend(loc="upper left", fontsize=9)
ax.set_ylabel("Hs (m)")
ax.set_title("Monthly means")
year_axis(axes[1])
fig.suptitle("Linear trend of Hs (seasonal cycle modelled with 3 harmonics)", weight="bold")
finish(fig, "B3a_hs_trend_daily_monthly.png", HS_PLACE)

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), layout="constrained")
for ax, tr_, name in [(axes[0], tr_mean, "Annual mean Hs"), (axes[1], tr_p95, "Annual 95th percentile of daily Hs")]:
    ax.plot(tr_["x"], tr_["y"], color="#888888", lw=1)
    ax.scatter(tr_["x"], tr_["y"], c=[POS if v > tr_["y"].mean() else NEG for v in tr_["y"]],
               s=60, edgecolor="white", zorder=3)
    xx = np.array([tr_["x"].min(), tr_["x"].max()])
    ax.plot(xx, tr_["sen_int"] + tr_["sen"] * xx, color=POS, lw=2.4)
    text_box(ax, f"Sen {tr_['sen'] * 100:+.2f} cm/yr "
                 f"[{tr_['sen_lo'] * 100:+.2f}, {tr_['sen_hi'] * 100:+.2f}]\n"
                 f"Mann–Kendall τ = {tr_['tau']:+.2f}, p = {tr_['mk_p']:.3f}\n"
                 f"change over the record {tr_['sen'] * (xx[1] - xx[0]) * 100:+.1f} cm")
    ax.set_title(name)
    ax.set_ylabel("Hs (m)")
    ax.set_xticks(tr_["x"][::3])
finish(fig, "B3b_hs_trend_annual.png", HS_PLACE)

trend_table = pd.DataFrame({
    "slope (cm/yr)": [res_d["slope"] * 100, res_m["slope"] * 100, tr_mean["sen"] * 100, tr_p95["sen"] * 100],
    "CI low": [res_d["ci"][0] * 100, res_m["ci"][0] * 100, tr_mean["sen_lo"] * 100, tr_p95["sen_lo"] * 100],
    "CI high": [res_d["ci"][1] * 100, res_m["ci"][1] * 100, tr_mean["sen_hi"] * 100, tr_p95["sen_hi"] * 100],
    "p": [res_d["p"], res_m["p"], tr_mean["mk_p"], tr_p95["mk_p"]],
}, index=["daily (harmonic regression)", "monthly (harmonic regression)",
          "annual mean (Sen / Mann-Kendall)", "annual P95 (Sen / Mann-Kendall)"])
trend_table.round(3).to_csv(FIG_DIR / "B3_trend_summary.csv")
trend_table.round(3)

# %% [markdown]
# ### Is the trend related to climate change?
#
# Read the table above with these points in mind:
#
# 1. **Size.** Express the slope relative to the mean (% per decade) and compare it with the
#    interannual variability: a change of a few cm per decade on a ~1 m mean is small next to
#    the year-to-year swings of ±5–10 cm.
# 2. **Robustness.** Daily and monthly regressions have many (correlated) samples; the annual
#    Sen/Mann–Kendall test, with ~21 nearly independent values, is the honest test. A slope
#    "significant" for daily data but not for annual means is not a robust trend.
# 3. **Record length.** 21 years is shorter than the 30-year minimum for a climate trend and is
#    comparable to the time scales of ENSO (2–7 yr) and the Pacific Decadal / Interdecadal
#    Pacific Oscillation (10–30 yr). The PDO/IPO switched phase around 1999 and again around
#    2014, so a 2000–2020 trend is dominated by **natural variability** and its sign can change
#    by shifting the window a few years.
# 4. **Data homogeneity.** ERA5 assimilates altimeter Hs, and the number of satellites changed
#    over 2000–2020 (ERS-2, Envisat, Jason-1/2/3, CryoSat-2, SARAL, Sentinel-3). Such changes can
#    create step-like artefacts of a few cm, the same order as the trend.
# 5. **Physical expectation.** Observed climate-related increases in wave height are concentrated
#    in the Southern Ocean (Young & Ribal 2019) and in global wave power (~0.4 %/yr, Reguero et
#    al. 2019). Tumaco's swell comes partly from there, but the signal is attenuated over the
#    long propagation path, and tropical eastern-Pacific trends are weak and not significant in
#    most studies. Projections (CMIP5/6) give only small (±5 %) end-of-century changes here.
#
# **Conclusion:** the trend obtained is a description of 2000–2020, not evidence of climate
# change. Attribution would need ≥ 30–40 years, a homogeneous dataset and removal of the
# ENSO/PDO signal (e.g. regression on climate indices before estimating the trend).

# %%
sig_annual = tr_mean["mk_p"] < 0.05
print(f"Monthly regression: {res_m['slope'] * 1000:+.1f} cm/decade "
      f"({res_m['slope'] * 10 / res_m['mean'] * 100:+.1f} % of the mean per decade), p = {res_m['p']:.3f}.")
print(f"Annual means (Mann-Kendall): p = {tr_mean['mk_p']:.3f} -> "
      f"{'significant' if sig_annual else 'NOT significant'} at 5 %.")
print(f"Interannual std of annual means: {ann_mean.std() * 100:.1f} cm; trend over the record: "
      f"{tr_mean['sen'] * (ann_mean.index[-1] - ann_mean.index[0]) * 100:+.1f} cm.")

# %% [markdown]
# ## B4. Detrended series, climatology and anomalies (task 14)
#
# 1. **Detrend**: the linear trend from the monthly harmonic regression is removed, keeping the
#    mean level.
# 2. **Climatology**: mean of each calendar month of the detrended series.
# 3. **Anomaly**: detrended series minus the climatology of its calendar month. What remains is
#    variability that is neither trend nor regular seasonal cycle: intraseasonal storms,
#    interannual (ENSO) and decadal signals.

# %%
t_m = decimal_year(hs_m.index, "monthly")
hs_det = (hs_m - res_m["slope"] * (t_m - res_m["t_mean"])).rename("hs_detrended")
g = hs_det.groupby(hs_det.index.month)
hs_clim = pd.DataFrame({"mean": g.mean(), "std": g.std(), "min": g.min(), "max": g.max(), "n": g.count()})
hs_anom = (hs_det - g.transform("mean")).rename("anomaly")

fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, sharey=True, layout="constrained")
axes[0].plot(xm, hs_m, color=LINE, lw=1.2, marker="o", ms=2.5, label="Monthly mean Hs")
axes[0].plot(xm, res_m["fit"], color=POS, lw=2.4, ls="--",
             label=f"Linear trend ({res_m['slope'] * 100:+.2f} cm/yr)")
axes[0].set_title("Original series", loc="left")
axes[1].plot(xm, hs_det, color="#2a9d8f", lw=1.2, marker="o", ms=2.5, label="Detrended monthly Hs")
axes[1].axhline(hs_det.mean(), color="#555555", ls="--", label=f"Mean {hs_det.mean():.3f} m")
axes[1].set_title("Detrended series (trend removed, mean kept)", loc="left")
for ax in axes:
    ax.set_ylabel("Hs (m)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
year_axis(axes[1])
finish(fig, "B4a_hs_detrended.png", HS_PLACE)

xx = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
ax.fill_between(xx, hs_clim["min"], hs_clim["max"], color=LINE, alpha=0.14, label="Min–max across years")
ax.fill_between(xx, hs_clim["mean"] - hs_clim["std"], hs_clim["mean"] + hs_clim["std"], color=LINE,
                alpha=0.32, label="±1 standard deviation")
ax.plot(xx, hs_clim["mean"], color=DARK, lw=3, marker="o", ms=7, label="Climatological mean")
for mth, c, off in [(int(hs_clim["mean"].idxmax()), POS, 14), (int(hs_clim["mean"].idxmin()), NEG, -22)]:
    ax.annotate(f"{MONTHS[mth - 1]} {hs_clim['mean'][mth]:.2f} m", (mth, hs_clim["mean"][mth]),
                xytext=(0, off), textcoords="offset points", ha="center", color=c, weight="bold")
ax.set_xticks(xx, MONTHS)
ax.set_ylabel("Hs (m)")
ax.legend(loc="upper left", frameon=False)
ax.set_title(f"Climatological monthly cycle of detrended Hs; seasonal range "
             f"{hs_clim['mean'].max() - hs_clim['mean'].min():.2f} m")
finish(fig, "B4b_hs_climatology.png", HS_PLACE)

sd = hs_anom.std()
fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
if phase is not None:
    shade_enso(ax, phase)
ax.bar(xm, hs_anom, width=25, color=np.where(hs_anom >= 0, POS, NEG), alpha=0.8, zorder=2)
ax.plot(xm, hs_anom.rolling(12, center=True, min_periods=10).mean(), color="black", lw=2.4,
        zorder=4, label="12-month running mean")
for k, ls in [(1, "--"), (2, ":")]:
    for sg in (1, -1):
        ax.axhline(sg * k * sd, color="#777777", lw=1, ls=ls, zorder=1)
ax.axhline(0, color="#333333", lw=0.8)
for idx_, c, va in [(hs_anom.idxmax(), POS, "bottom"), (hs_anom.idxmin(), NEG, "top")]:
    ax.annotate(f"{idx_:%b %Y}\n{hs_anom[idx_]:+.2f} m", (idx_ + pd.Timedelta(days=14), hs_anom[idx_]),
                xytext=(0, 6 if va == "bottom" else -6), textcoords="offset points", ha="center",
                va=va, color=c, fontsize=9, weight="bold")
ax.set_ylabel("Hs anomaly (m)")
ax.legend(loc="upper left", ncol=3, fontsize=9)
text_box(ax, f"σ = {sd:.3f} m ({sd / hs_det.mean():.1%} of the mean)\n"
             f"lag-1 autocorrelation {lag1(hs_anom):.2f}\n"
             f"months beyond ±2σ: {(hs_anom.abs() > 2 * sd).mean():.1%}")
year_axis(ax)
ax.set_title("Monthly Hs anomaly (detrended, seasonal cycle removed); dashed ±1σ, dotted ±2σ")
finish(fig, "B4c_hs_anomaly.png", HS_PLACE)

# %% [markdown]
# **Anomaly by ENSO phase.** If ENSO modulates the wave climate, months in El Niño and La Niña
# should differ in their mean anomaly. Consecutive months are not independent, so the 95 %
# intervals and the El Niño – La Niña test use an effective sample size.

# %%
phase_stats = {}
if phase is not None:
    dfp = pd.DataFrame({"anom": hs_anom, "phase": phase}).dropna()
    r1a = lag1(hs_anom)
    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    rng = np.random.default_rng(0)
    for i, (code, name, c) in enumerate([(-1, "La Niña", NEG), (0, "Neutral", "#8f8f8f"), (1, "El Niño", POS)]):
        v = dfp.loc[dfp.phase == code, "anom"].values
        if len(v) < 3:
            continue
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=c, alpha=0.25, edgecolor=c), medianprops=dict(color=c, lw=2))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=14, color=c, alpha=0.7, zorder=3)
        ne = n_eff_ar1(len(v), r1a)
        half_ci = stats.t.ppf(0.975, max(ne - 1, 1)) * v.std(ddof=1) / np.sqrt(ne)
        ax.errorbar(i, v.mean(), yerr=half_ci, fmt="D", color="black", ms=8, capsize=8, lw=2, zorder=5)
        phase_stats[code] = (v.mean(), v.std(ddof=1), len(v), ne)
        ax.text(i, 1.01, f"n = {len(v)}\nmean {v.mean() * 100:+.1f} cm", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=c, weight="bold", fontsize=9)
    extra = ""
    if 1 in phase_stats and -1 in phase_stats:
        (m1, s1, _, e1), (m2, s2, _, e2) = phase_stats[1], phase_stats[-1]
        se = np.sqrt(s1 ** 2 / e1 + s2 ** 2 / e2)
        dof = se ** 4 / ((s1 ** 2 / e1) ** 2 / (e1 - 1) + (s2 ** 2 / e2) ** 2 / (e2 - 1))
        p_en = 2 * stats.t.sf(abs((m1 - m2) / se), dof)
        extra = f"; El Niño − La Niña = {(m1 - m2) * 100:+.1f} cm, p = {p_en:.2f}"
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(range(3), ["La Niña", "Neutral", "El Niño"])
    ax.set_ylabel("Monthly Hs anomaly (m)")
    ax.set_title(f"Hs anomaly by ENSO phase (ONI){extra}\n◆ mean with 95 % CI (effective n)", pad=40)
    finish(fig, "B4d_hs_anomaly_by_enso.png", HS_PLACE)

# %% [markdown]
# ### What the anomaly series shows
# * **Size**: σ of the monthly anomaly is a few % of the mean Hs, much smaller than the seasonal
#   range (B4b): the wave climate at Tumaco is dominated by the regular seasonal cycle of Southern
#   Ocean swell, and is otherwise mild.
# * **Persistence**: a positive lag-1 autocorrelation and the 12-month running mean show that
#   anomalies come in multi-month spells rather than as independent months; this is the signature
#   of large-scale climate modes rather than of individual storms.
# * **ENSO**: compare the spells with the El Niño / La Niña shading and with B4d. Strong El Niño
#   winters (e.g. 2009–10, 2015–16) tend to bring more energetic North Pacific swell, while La
#   Niña is associated with weaker swell. The *p*-value in B4d says whether that difference is
#   distinguishable from noise with the effective sample size available.
# * **Extremes**: the largest monthly anomalies point to individual swell events; their dates can
#   be checked against known storms.

# %% [markdown]
# ## B5–B6. FFT, Welch periodogram and comparison (tasks 15 and 16)
#
# Both spectra use the **last 20 years (240 months)**, as the task asks. Mean and linear trend are
# removed first so they do not leak into the lowest frequencies.
#
# * **FFT periodogram**: whole 240-month record, frequency resolution 1/20 cycles per year, but
#   each value has only 2 degrees of freedom (very noisy).
# * **Welch**: 8-year Hann-windowed segments with 50 % overlap (4 segments). Averaging reduces
#   the variance (more degrees of freedom) at the cost of resolution (1/8 cycles per year): the
#   2–4 yr ENSO band is sampled at periods of 4, 2.7 and 2 years.
# * **Significance**: against an AR(1) red-noise background. Correction with respect to the
#   previous version: the AR(1) parameters are now estimated from the **anomaly** in both cases.
#   Fitting AR(1) to the raw monthly series uses a lag-1 autocorrelation dominated by the
#   deterministic annual cycle, which inflates the background at low frequencies and biases the
#   test against the ENSO band. The null hypothesis is "persistent random noise around the
#   seasonal cycle", which is the same for both series, so the two spectra are tested against the
#   same background.

# %%
nwin = SPECTRAL_YEARS * 12
hs_m20, hs_anom20 = hs_m.iloc[-nwin:], hs_anom.iloc[-nwin:]


def fft_periodogram(x, fs=FS):
    """One-sided periodogram, scaled so sum(P) * df = variance (Parseval)."""
    n = len(x)
    F = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1 / fs)
    P = np.abs(F) ** 2 / (fs * n)
    P[1:] *= 2
    if n % 2 == 0:
        P[-1] /= 2
    return f[1:], P[1:]


def red_noise(f, var, r1, fs=FS):
    """One-sided spectrum of an AR(1) process with variance var and lag-1 r1."""
    return 2 * var * (1 - r1 ** 2) / (fs * (1 - 2 * r1 * np.cos(2 * np.pi * f / fs) + r1 ** 2))


def spectrum(series, noise_from):
    x = signal.detrend(series.values.astype(float), type="linear")
    xa = signal.detrend(noise_from.values.astype(float), type="linear")
    ff, Pf = fft_periodogram(x)
    nper = int(WELCH_SEGMENT_YEARS * FS)
    fw, Pw = signal.welch(x, fs=FS, window="hann", nperseg=nper, noverlap=nper // 2,
                          detrend="constant")
    fw, Pw = fw[1:], Pw[1:]
    k = 1 + (len(x) - nper) // (nper // 2)
    edof = 36 * k ** 2 / (19 * k - 1)                    # Hann, 50 % overlap (Welch 1967)
    r1 = lag1(xa)
    red = red_noise(fw, xa.var(), r1)
    sig = red * stats.chi2.ppf(0.95, edof) / edof
    peaks = [i for i in range(len(fw)) if Pw[i] > sig[i]
             and (i == 0 or Pw[i] >= Pw[i - 1]) and (i == len(fw) - 1 or Pw[i] >= Pw[i + 1])]
    return dict(ff=ff, Pf=Pf, fw=fw, Pw=Pw, red=red, sig=sig, r1=r1, k=k, edof=edof,
                peaks=peaks, var=x.var(), parseval=Pf.sum() * (ff[1] - ff[0]) / x.var(),
                t0=series.index[0], t1=series.index[-1])


def variance_budget(spec, band=ENSO_BAND):
    """Share of variance by frequency group, integrating the FFT periodogram."""
    f, P = spec["ff"], spec["Pf"]
    df = f[1] - f[0]
    harm = np.isclose(f % 1, 0, atol=df / 2) | np.isclose(f % 1, 1, atol=df / 2)
    per = 1 / f
    enso = (per >= band[0]) & (per <= band[1]) & ~harm
    low = (per > band[1]) & ~harm
    tot = P.sum()
    return {"Annual cycle + harmonics": P[harm].sum() / tot,
            f"ENSO band {band[0]:g}–{band[1]:g} yr": P[enso].sum() / tot,
            f"Low frequency > {band[1]:g} yr": P[low].sum() / tot,
            f"Intra-annual / other < {band[0]:g} yr": P[~(harm | enso | low)].sum() / tot}


def spectrum_axes(ax, fmin):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(fmin * 0.9, FS / 2 * 1.05)
    ax.axvspan(1 / ENSO_BAND[1], 1 / ENSO_BAND[0], color=BAND_C, alpha=0.7, lw=0, zorder=0,
               label=f"ENSO band ({ENSO_BAND[0]:g}–{ENSO_BAND[1]:g} yr)")
    for k, lab in [(1, "12 mo"), (2, "6 mo"), (3, "4 mo")]:
        ax.axvline(k, color="#b5b5b5", lw=1, zorder=1)
        ax.text(k, 0.985, f" {lab}", transform=ax.get_xaxis_transform(), va="top", fontsize=9, color="#777777")
    ax.set_xlabel("Frequency (cycles per year)")
    ax.set_ylabel("PSD (m² · yr)")
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.25, 0.5, 1, 2, 3, 6]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    inv = lambda v: 1 / np.where(np.isclose(v, 0), np.inf, v)  # noqa: E731
    sec = ax.secondary_xaxis("top", functions=(inv, inv))
    sec.set_xlabel("Period (years)")
    sec.xaxis.set_major_locator(FixedLocator([20, 8, 4, 2, 1, 0.5, 0.25]))
    sec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    sec.xaxis.set_minor_locator(NullLocator())


def plot_spectrum(spec, label, color, fname):
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    spectrum_axes(ax, spec["ff"].min())
    ax.plot(spec["ff"], spec["Pf"], color="#b3b3b3", lw=1, label="FFT periodogram (2 dof per value)")
    ax.plot(spec["fw"], spec["Pw"], color=color, lw=2.6, marker="o", ms=5,
            label=f"Welch ({WELCH_SEGMENT_YEARS:g}-yr Hann segments, 50 % overlap, K = {spec['k']})")
    ax.plot(spec["fw"], spec["red"], color="#444444", ls="--", lw=1.6,
            label=f"AR(1) background of the anomaly (r1 = {spec['r1']:.2f})")
    ax.plot(spec["fw"], spec["sig"], color=POS, ls=":", lw=2, label=f"95 % level (EDOF ≈ {spec['edof']:.1f})")
    for i in spec["peaks"]:
        ax.plot(spec["fw"][i], spec["Pw"][i], marker="*", ms=17, color="#ffb000", mec="black", ls="none", zorder=6)
        ax.annotate(fmt_period(1 / spec["fw"][i]), (spec["fw"][i], spec["Pw"][i]), xytext=(0, 12),
                    textcoords="offset points", ha="center", weight="bold")
    lo_ = min(spec["Pw"].min(), spec["red"].min()) / 30
    ax.set_ylim(lo_, max(spec["Pf"].max(), spec["Pw"].max()) * 8)
    ax.legend(loc="lower left", fontsize=9)
    n_above = int((spec["Pw"] > spec["sig"]).sum())
    text_box(ax, f"{n_above} of {len(spec['fw'])} Welch frequencies above the 95 % line\n"
                 f"(≈{0.05 * len(spec['fw']):.1f} expected by chance alone)", "upper left")
    ax.set_title(f"Spectrum of {label}, {spec['t0']:%Y-%m} to {spec['t1']:%Y-%m}")
    finish(fig, fname, HS_PLACE)


sp_m = spectrum(hs_m20, hs_anom20)
sp_a = spectrum(hs_anom20, hs_anom20)
plot_spectrum(sp_m, "monthly mean Hs (task 15)", LINE, "B5_spectrum_monthly_hs.png")

# %% [markdown]
# **Reading the monthly-Hs periodogram (task 15).** Expect the dominant, significant peak at
# 12 months (starred): it is the annual cycle of Southern Ocean swell and local winds; harmonics at 6 and
# 4 months appear because that cycle is not a pure sinusoid (B4b). Their power is one to two
# orders of magnitude above everything else, so they hide the weaker interannual (ENSO)
# variability in the lowest frequencies. A peak counts as real only if it rises above the dotted
# 95 % line, and even then about 5 % of the frequencies are expected to cross it by chance.

# %%
plot_spectrum(sp_a, "the monthly Hs anomaly (task 16)", ANOM_C, "B6a_spectrum_anomaly.png")

bm, ba = variance_budget(sp_m), variance_budget(sp_a)
fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
spectrum_axes(ax, sp_m["fw"].min())
ax.plot(sp_m["fw"], sp_m["Pw"], color=LINE, lw=2.6, marker="o", ms=5, label="Monthly Hs (task 15)")
ax.plot(sp_a["fw"], sp_a["Pw"], color=ANOM_C, lw=2.6, marker="s", ms=5, label="Hs anomaly (task 16)")
ax.plot(sp_m["fw"], sp_m["sig"], color="#555555", lw=1.4, ls=":", label="95 % level (common AR(1) background)")
i1 = int(np.argmin(np.abs(sp_a["fw"] - 1)))
ax.annotate("annual cycle removed\nin the anomaly", (sp_a["fw"][i1], sp_a["Pw"][i1]),
            xytext=(sp_a["fw"][i1] * 1.9, sp_a["Pw"][i1] * 0.2), ha="center", color="#7a3b12",
            arrowprops=dict(arrowstyle="->", color="#7a3b12"))
lo_ = ax.get_ylim()
ax.set_ylim(lo_[0], lo_[1] * 4)
ax.legend(loc="lower left", fontsize=9)
w = max(len(n) for n in bm)
lines = [f"{'Share of variance':<{w}}  Monthly  Anomaly"]
lines += [f"{n:<{w}}  {bm[n]:7.1%}  {ba[n]:7.1%}" for n in bm]
lines.append(f"{'Total variance (m2)':<{w}}  {sp_m['var']:7.4f}  {sp_a['var']:7.4f}")
text_box(ax, "\n".join(lines), "upper left", fontsize=8.5, family="monospace")
ax.set_title("Welch periodograms compared: monthly Hs vs Hs anomaly")
finish(fig, "B6b_spectrum_comparison.png", HS_PLACE)
(pd.DataFrame({"monthly Hs": bm, "anomaly": ba}) * 100).round(1).rename_axis("% of variance")

# %% [markdown]
# **Comparison (task 16).**
# * Removing the climatology deletes the 12-, 6- and 4-month peaks: in the anomaly they drop to
#   the background level. The total variance falls accordingly (table), which shows how much of
#   the wave climate is just the seasonal cycle.
# * Away from the annual harmonics the two Welch curves almost coincide: removing a 12-value
#   climatology only affects exactly 1, 2, 3 … cycles per year (and, through window leakage,
#   their immediate neighbours). The interannual part of the spectrum is the **same** in both.
# * What changes is visibility: in the anomaly the low-frequency (> 1 yr) variability, including
#   the 2–4 yr ENSO band, becomes the main feature instead of being two orders of magnitude below
#   the annual peak. Whether it is **significant** depends on whether it clears the red-noise
#   95 % line: with only 20 years (~5–10 ENSO cycles) and 4 Welch segments, interannual peaks are
#   hard to separate from red noise, which is why B7 tests the ENSO link directly against the ONI.

# %% [markdown]
# ## B7. Hs from ENSO frequencies only: 2–4 year band-pass (task 17)
#
# * **Filter**: Butterworth band-pass between 1/4 and 1/2 cycles per year (order 2), applied
#   forwards and backwards (`sosfiltfilt`) so there is **no phase shift**: ENSO-band maxima stay
#   at their true dates. The double pass squares the response: gain 0.5 at 2 and 4 years, ≈ 1 at
#   ~3 years (figure B7b).
# * **Input**: the full (21-year) anomaly series. Band-passing the raw monthly Hs gives virtually
#   the same result because the filter already rejects the annual cycle (checked below).
# * **Edges**: near both ends the output depends on padding rather than data; that zone (from the
#   width of the filter's impulse response) is hatched and excluded from statistics.
# * **ENSO-only Hs** = mean Hs + band-passed anomaly.

# %%
sos = signal.butter(FILTER_ORDER, [1 / ENSO_BAND[1], 1 / ENSO_BAND[0]], btype="bandpass", fs=FS, output="sos")


def band_pass(x):
    x = np.asarray(x, float)
    return signal.sosfiltfilt(sos, x, padtype="odd", padlen=min(len(x) - 1, int(3 * ENSO_BAND[1] * FS)))


def edge_width(n=1201):
    """e-folding half-width (months) of the zero-phase impulse response envelope."""
    imp = np.zeros(n)
    imp[n // 2] = 1
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, imp)))
    return int(np.max(np.abs(np.where(env / env.max() >= np.exp(-1))[0] - n // 2)))


edge = edge_width()
enso_anom = pd.Series(band_pass(hs_anom.values), index=hs_anom.index, name="enso_band")
enso_hs = (hs_det.mean() + enso_anom).rename("hs_enso")
interior = pd.Series(False, index=hs_anom.index)
interior.iloc[edge:len(hs_anom) - edge] = True
var_frac = enso_anom[interior].var() / hs_anom[interior].var()
raw_bp = band_pass(hs_m.values - hs_m.mean())
print(f"edge zone: {edge} months at each end")
print(f"ENSO band carries {var_frac:.0%} of the anomaly variance (interior); "
      f"std {enso_anom[interior].std() * 100:.1f} cm, range {enso_hs[interior].min():.3f}–"
      f"{enso_hs[interior].max():.3f} m")
print(f"rms difference, band-pass of raw monthly Hs vs of anomaly (interior): "
      f"{np.sqrt(np.mean((raw_bp - enso_anom.values)[interior.values] ** 2)) * 100:.2f} cm")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, layout="constrained",
                         gridspec_kw={"height_ratios": [1, 1.25]})
ax = axes[0]
ax.plot(xm, hs_anom, color="#a0a0a0", lw=1, label="Monthly Hs anomaly")
ax.plot(xm, enso_anom, color="black", lw=2.4, label="2–4 yr component")
ax.axhline(0, color="#555555", lw=0.8)
ax.set_ylabel("Anomaly (m)")
ax.legend(loc="upper left", ncol=2)
ax.set_title(f"Same scale: the ENSO band carries {var_frac:.0%} of the anomaly variance", loc="left")
ax = axes[1]
if phase is not None:
    shade_enso(ax, phase)
ax.fill_between(xm, hs_det.mean(), enso_hs, where=enso_anom >= 0, color=POS, alpha=0.5, interpolate=True, lw=0)
ax.fill_between(xm, hs_det.mean(), enso_hs, where=enso_anom < 0, color=NEG, alpha=0.5, interpolate=True, lw=0)
ax.plot(xm, enso_hs, color="black", lw=2, label="Hs from ENSO frequencies only")
ax.axhline(hs_det.mean(), color="#555555", lw=0.8)
for a, b in [(xm[0], xm[edge]), (xm[-1 - edge], xm[-1])]:
    ax.axvspan(a, b, facecolor="none", hatch="///", edgecolor="#9a9a9a", lw=0, zorder=4)
h, l_ = ax.get_legend_handles_labels()
h.append(Patch(facecolor="none", hatch="///", edgecolor="#9a9a9a"))
l_.append(f"Edge zone ({edge} months)")
ax.legend(h, l_, loc="upper left", ncol=2, fontsize=9)
ax.set_ylabel("Hs (m)")
ax.set_title("ENSO-band Hs: mean + 2–4 yr component", loc="left")
year_axis(axes[1])
finish(fig, "B7a_hs_enso_band.png", HS_PLACE)

f_r, H = signal.sosfreqz(sos, worN=8192, fs=FS)
gain = np.abs(H[1:]) ** 2
fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
ax.axvspan(*ENSO_BAND, color=BAND_C, alpha=0.8, lw=0, label="Target band")
ax.plot(1 / f_r[1:], gain, color="black", lw=2.6, label="Amplitude gain (forward + backward)")
ax.axhline(0.5, color="#888888", ls="--", lw=1)
for T in (1, 1.5, 3, 6, 8):
    gT = float(np.interp(1 / T, f_r[1:], gain))
    ax.plot(T, gT, "o", color=POS)
    ax.annotate(f"{T:g} yr: {gT:.2f}", (T, gT), xytext=(6, 6), textcoords="offset points", fontsize=9, color=POS)
ax.set_xscale("log")
ax.set_xlim(0.4, 20)
ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3, 4, 6, 8, 16]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Period (years)")
ax.set_ylabel("Gain")
ax.legend(loc="upper right")
ax.set_title(f"Frequency response of the band-pass filter (Butterworth order {FILTER_ORDER}, zero phase)")
finish(fig, "B7b_filter_response.png")

# %% [markdown]
# **Does the ENSO-band Hs actually follow ENSO?** The ONI is band-passed with the same filter
# (over its full 1950–present record, so it has no edge effect) and correlated with the
# ENSO-band Hs at lags of ±24 months. Band-passed series are very smooth, so the number of
# independent values is small; significance uses the effective sample size of Pyper & Peterman
# (1998). Without this check, the band-pass output is only "variability at ENSO time scales",
# not necessarily "variability caused by ENSO".

# %%
def n_effective(x, y):
    """Effective number of independent pairs (Pyper & Peterman 1998)."""
    x, y = np.asarray(x) - np.mean(x), np.asarray(y) - np.mean(y)
    n = len(x)
    s = sum((n - j) / n * np.corrcoef(x[:-j], x[j:])[0, 1] * np.corrcoef(y[:-j], y[j:])[0, 1]
            for j in range(1, n // 5 + 1))
    return float(np.clip(1 / (1 / n + 2 / n * s), 3, n))


def r_critical(ne, alpha=0.05):
    tc = stats.t.ppf(1 - alpha / 2, ne - 2)
    return float(tc / np.sqrt(ne - 2 + tc ** 2))


if oni is not None:
    oni_bp = pd.Series(band_pass(oni.values - oni.mean()), index=oni.index).reindex(hs_anom.index)
    lags = range(-24, 25)                                # positive: Hs lags the ONI
    cc = pd.Series({k: enso_anom[interior & oni_bp.shift(k).notna()]
                    .corr(oni_bp.shift(k)[interior & oni_bp.shift(k).notna()]) for k in lags})
    k_best = int(cc.abs().idxmax())
    ok = interior & oni_bp.shift(k_best).notna()
    ne = n_effective(enso_anom[ok], oni_bp.shift(k_best)[ok])
    rc = r_critical(ne)
    r_raw = hs_anom.corr(oni.reindex(hs_anom.index))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), layout="constrained")
    z = lambda s: (s - s[interior].mean()) / s[interior].std()   # noqa: E731
    axes[0].plot(xm, z(oni_bp), color="#e39b2d", lw=2.4, label="ONI, same band-pass")
    axes[0].plot(xm, z(enso_anom), color="black", lw=2.4, label="ENSO-band Hs")
    for a, b in [(xm[0], xm[edge]), (xm[-1 - edge], xm[-1])]:
        axes[0].axvspan(a, b, facecolor="none", hatch="///", edgecolor="#9a9a9a", lw=0)
    axes[0].axhline(0, color="#555555", lw=0.8)
    axes[0].set_ylabel("Standardised units")
    axes[0].legend(loc="upper left", ncol=2)
    year_axis(axes[0])
    axes[0].set_title("Both series band-passed identically (hatched: Hs edge zone, excluded)", loc="left")
    ax = axes[1]
    ax.axhspan(-rc, rc, color="#eeeeee", lw=0, label=f"not significant (|r| < {rc:.2f}, N_eff = {ne:.0f})")
    ax.plot(cc.index, cc.values, color="black", lw=2.4, marker="o", ms=4)
    ax.plot(k_best, cc[k_best], "o", ms=13, mfc="none", mec=POS, mew=2.5)
    ax.annotate(f"r = {cc[k_best]:+.2f} at {k_best:+d} months", (k_best, cc[k_best]),
                xytext=(0, 16 if cc[k_best] > 0 else -24), textcoords="offset points", ha="center",
                color=POS, weight="bold")
    ax.axvline(0, color="#555555", lw=0.8)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlim(-24.5, 24.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Lag (months); positive = Hs lags the ONI")
    ax.set_ylabel("Correlation r")
    ax.legend(loc="upper left")
    ax.set_title("Lagged correlation, band-passed Hs vs band-passed ONI", loc="left")
    finish(fig, "B7c_enso_band_vs_oni.png", HS_PLACE)
    verdict = "significant" if abs(cc[k_best]) > rc else "NOT significant"
    print(f"Best lag {k_best:+d} months: r = {cc[k_best]:+.2f} ({verdict}; threshold {rc:.2f}, N_eff {ne:.1f}).")
    print(f"Unfiltered anomaly vs ONI at lag 0: r = {r_raw:+.2f}")

# %% [markdown]
# ### Interpretation of the ENSO-band series
# * The band-passed series isolates oscillations of 2–4 years; its amplitude (a few cm) and its
#   share of the anomaly variance quantify how much of the non-seasonal wave climate at Tumaco
#   occurs at ENSO time scales.
# * A correlation that is significant with the effective sample size, at a physically plausible
#   lag of a few months, supports an ENSO teleconnection: ENSO changes the North Pacific storm
#   track (boreal-winter NW swell) and the cross-equatorial winds of the Panama Bight, and the
#   wave response lags the central-Pacific SST signal measured by the ONI.
# * With 21 years only ~5–8 ENSO cycles are sampled, the edges are unreliable, and the 2–4 yr
#   window misses longer events (ENSO periods span 2–7 yr). A non-significant result is
#   therefore "not detectable with this record", not "no relationship".

# %%
# --------------------------------------------------------------- export ----
out = pd.DataFrame({"hs": hs_m, "hs_detrended": hs_det, "anomaly": hs_anom,
                    "enso_band_anomaly": enso_anom, "hs_enso_only": enso_hs, "edge_zone": ~interior})
if oni is not None:
    out["oni"] = oni.reindex(hs_m.index)
    out["enso_phase"] = phase
out.round(5).to_csv(FIG_DIR / "B_monthly_hs_series.csv")
hs_clim.round(4).to_csv(FIG_DIR / "B4_hs_climatology.csv")
print(f"Figures and tables written to {FIG_DIR.resolve()}")
