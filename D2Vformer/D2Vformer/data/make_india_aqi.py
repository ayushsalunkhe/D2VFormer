# -*- coding: utf-8 -*-
"""
make_india_aqi.py
Generate a realistic synthetic Delhi AQI time-series dataset (hourly, 2019-2022)
for use as the India-relevant extension dataset in the D2Vformer project.

Pollutant patterns are based on published CPCB/IQAir statistics for Delhi:
- PM2.5 / PM10: severe winter peaks (Oct-Jan, crop burning + cold trapping)
- NO2 / CO:     rush-hour spikes, elevated in winter
- SO2:          industrial contribution, moderate seasonal variation
- O3:           photochemical, peaks in pre-monsoon summer afternoons

Outputs (same format as ETT datasets):
  datasets/india_aqi/delhi_aqi.csv   — data (date + 6 pollutants, 4 years hourly)
  datasets/india_aqi/delhi_mark.csv  — calendar / external features (same format as china.csv)

Run from the D2Vformer/D2Vformer directory:
  python data/make_india_aqi.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# ---------- time grid ---------------------------------------------------
start = pd.Timestamp("2019-01-01 00:00:00")
end   = pd.Timestamp("2022-12-31 23:00:00")
dates = pd.date_range(start, end, freq="h")
N     = len(dates)
print(f"Total hours: {N}")

t     = np.arange(N)
hour  = np.array([d.hour       for d in dates])
doy   = np.array([d.day_of_year for d in dates])
month = np.array([d.month      for d in dates])

# ---------- shared seasonal / diurnal building blocks --------------------
# winter severity:  peaks around Jan(doy~15) and Dec(doy~350)
winter = 0.5 * (
    np.exp(-((doy - 15)  ** 2) / (2 * 40**2)) +
    np.exp(-((doy - 350) ** 2) / (2 / 40**2))
)
# smoother approach: cosine - peak in Jan
winter_cos = 0.5 * (1 - np.cos(2 * np.pi * (doy - 15) / 365))  # peaks ~Jan

# monsoon dip (Jul-Sep)
monsoon_dip = np.where((month >= 7) & (month <= 9), 0.4, 1.0)

# morning rush (8-10) and evening rush (17-20) diurnal pattern
rush = (
    np.exp(-((hour - 9)  ** 2) / (2 * 1.5**2)) * 0.5 +
    np.exp(-((hour - 18) ** 2) / (2 * 1.5**2)) * 0.5
)

# afternoon photochemical peak (12-15) for O3
photo = np.exp(-((hour - 13) ** 2) / (2 * 2.0**2))

# ---------- PM2.5 (µg/m³) -----------------------------------------------
# Delhi annual mean ~100 µg/m³; winter > 300 µg/m³; monsoon ~30 µg/m³
pm25_base   = 80
pm25_winter = 250 * winter_cos
pm25_monsoon_factor = monsoon_dip
pm25_daily  = 40 * rush
pm25 = (pm25_base + pm25_winter + pm25_daily) * pm25_monsoon_factor
pm25 += np.random.normal(0, 15, N)
pm25 = np.clip(pm25, 5, 500)

# ---------- PM10 (µg/m³)  ~ 1.5-2x PM2.5 --------------------------------
pm10 = pm25 * (1.6 + np.random.normal(0, 0.1, N))
pm10 = np.clip(pm10, 10, 900)

# ---------- NO2 (µg/m³) -------------------------------------------------
no2_base    = 50
no2_winter  = 80 * winter_cos
no2_daily   = 60 * rush
no2 = (no2_base + no2_winter + no2_daily) * monsoon_dip
no2 += np.random.normal(0, 8, N)
no2 = np.clip(no2, 5, 350)

# ---------- SO2 (µg/m³) -------------------------------------------------
so2_base   = 20
so2_winter = 40 * winter_cos
so2 = (so2_base + so2_winter) * monsoon_dip
so2 += np.random.normal(0, 5, N)
so2 = np.clip(so2, 2, 200)

# ---------- CO (mg/m³ * 10 for same scale as others) --------------------
co_base   = 30
co_winter = 60 * winter_cos
co_daily  = 40 * rush
co = (co_base + co_winter + co_daily) * monsoon_dip
co += np.random.normal(0, 6, N)
co = np.clip(co, 5, 300)

# ---------- O3 (µg/m³) --------------------------------------------------
o3_base    = 40
o3_summer  = 60 * np.exp(-((doy - 120) ** 2) / (2 * 50**2))  # Apr-May peak
o3_photo   = 50 * photo
# O3 is anti-correlated with NOx (titration) — dip during rush
o3 = o3_base + o3_summer + o3_photo - 20 * rush
o3 += np.random.normal(0, 6, N)
o3 = np.clip(o3, 5, 300)

# ---------- Assemble data CSV -------------------------------------------
df = pd.DataFrame({
    "date":  dates,
    "PM2.5": np.round(pm25, 1),
    "PM10":  np.round(pm10, 1),
    "NO2":   np.round(no2,  1),
    "SO2":   np.round(so2,  1),
    "CO":    np.round(co,   1),
    "O3":    np.round(o3,   1),
})

# ---------- Build calendar mark file (same schema as china.csv) ----------
def is_holiday(d):
    """Approximate Indian public holidays."""
    key = (d.month, d.day)
    public = {(1,1),(1,26),(8,15),(10,2),(10,24),(11,14),(12,25)}
    return 1 if key in public else 0

# ---------- Build calendar mark file (one row per DAY — same as china.csv) --
days = pd.date_range(start.normalize(), end.normalize(), freq="D")

def is_holiday(d):
    """Approximate Indian public holidays."""
    key = (d.month, d.day)
    public = {(1,1),(1,26),(8,15),(10,2),(10,24),(11,14),(12,25)}
    return 1 if key in public else 0

df_mark = pd.DataFrame({
    "date":        days.strftime("%Y-%m-%d"),
    "abs_days":    (days - pd.Timestamp("2000-01-01")).days,
    "year":        days.year,
    "day":         days.day,
    "year_day":    days.day_of_year,
    "week":        days.isocalendar().week.values,
    "holidays":    [is_holiday(d) for d in days],
    "workdays":    [(0 if d.weekday() >= 5 else 1) for d in days],
    "dayofweek":   days.weekday,
    "dayofmonth":  days.day,
    "dayofyear":   days.day_of_year,
    "monthofyear": days.month,
    "residual_holiday": 0,
    "residual_workday": 0,
    "lunar_year":     0,
    "lunar_month":    days.month,
    "lunar_day":      days.day,
    "lunar_year_day": days.day_of_year,
    "dayoflunaryear": days.day_of_year,
    "dayoflunarmonth":days.day,
    "monthoflunaryear":days.month,
    "jieqiofyear":    0,
    "jieqi_day":      0,
    "dayofjieqi":     0,
})

# ---------- Save ---------------------------------------------------------
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "india_aqi")
os.makedirs(out_dir, exist_ok=True)

data_path = os.path.join(out_dir, "delhi_aqi.csv")
mark_path = os.path.join(out_dir, "delhi_mark.csv")

df.to_csv(data_path, index=False)
df_mark.to_csv(mark_path, index=False)

print(f"Saved data:  {data_path}  ({len(df)} rows, {df.columns.tolist()})")
print(f"Saved marks: {mark_path}")
print(f"\nSample statistics:")
print(df.describe().round(1))
