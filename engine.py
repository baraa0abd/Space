"""Decision engine (Mosul / Nineveh and San Francisco Bay Area): place + date/time + observation type ->
Sky Score, class, best time, best spots. Every public function takes region="mosul" | "sf".

  Sky Score (0-100) = weighted sum of four component scores, weights depend on the observation type:
    light pollution  - hybrid: XGBoost sky-quality model + calibrated physics sky glow, on recent NASA VNP46A2 lights
    weather          - cloud, humidity, dust (AOD), visibility from the Open-Meteo / CAMS forecast
    moon             - illumination and altitude (astropy)
    elevation        - Open-Meteo elevation (higher = less haze above you)
    smoke (SF)       - PM2.5 from CAMS (wildfire smoke); fog (SF) = low cloud, ignored for sites above the marine layer
  then hard gates: daylight/twilight and overcast cap the score. Classes: 0-25 Poor, 26-50 Moderate, 51-75 Good, 76-100 Excellent.
"""
import io
import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.ndimage import distance_transform_edt

import skyclass
import skyq

DATA = skyq.DATA
NINEVEH = skyq.NINEVEH
STEP = 2  # grid cells every 2 px = 0.04 deg (~4 km)
CELLS_PATH = os.path.join(DATA, "nineveh_cells.csv")
VNP_PATH = os.path.join(DATA, "vnp46a2_iraqwork.npy")
VNP_LAYER = "VIIRS_SNPP_GapFilled_BRDF_Corrected_DayNightBand_Radiance"  # NASA Black Marble VNP46A2 (daily)

TYPES = {  # key: (Arabic label, weights light/weather/moon/elevation)
    "stargazing": ("مشاهدة النجوم ✨", (0.40, 0.35, 0.15, 0.10)),
    "milky_way": ("درب التبانة 🌌", (0.45, 0.25, 0.25, 0.05)),
    "astrophotography": ("التصوير الفلكي 📷", (0.40, 0.30, 0.25, 0.05)),
    "meteors": ("رصد الشهب ☄️", (0.35, 0.35, 0.25, 0.05)),
}
BINS = [25, 50, 75]  # -> Poor / Moderate / Good / Excellent
SF_WEIGHTS = {  # plans/sf_usa_3day_plan.json: light / weather / moon / elevation / smoke
    "stargazing": (0.35, 0.30, 0.15, 0.10, 0.10), "milky_way": (0.40, 0.20, 0.25, 0.05, 0.10),
    "astrophotography": (0.35, 0.25, 0.25, 0.05, 0.10), "meteors": (0.30, 0.30, 0.25, 0.05, 0.10),
}


def _sf_cells():
    import sf  # lazy: sf imports engine
    return sf.cells()


REGIONS = {
    "mosul": dict(bbox=NINEVEH, tz="Asia/Baghdad", cells=lambda: cells(), fog=False, smoke=False,
                  weights={k: w + (0.0,) for k, (_, w) in TYPES.items()}),
    "sf": dict(bbox=(36.4, -123.3, 38.6, -121.0), tz="America/Los_Angeles", cells=_sf_cells, fog=True, smoke=True,
               weights=SF_WEIGHTS),  # bbox = sf.STUDY
}


def in_region(lat, lon, region):
    a0, o0, a1, o1 = REGIONS[region]["bbox"]
    return a0 <= lat <= a1 and o0 <= lon <= o1


def to_class(score):
    return np.digitize(np.asarray(score), BINS, right=True)


# ---------------------------------------------------------------- recent satellite lights (VNP46A2)
def fetch_vnp46a2(days=48, every=6):
    """Median of recent daily VNP46A2 night-light images over the Iraq working box (NASA GIBS), cached.
    GIBS serves it as a greyscale PNG (brightness increases with radiance)."""
    if os.path.exists(VNP_PATH):
        return np.load(VNP_PATH)
    la0, lo0, la1, lo1 = skyclass.IRAQ_WORK
    end = date.today() - timedelta(days=2)
    stack = []
    for d in pd.date_range(end - timedelta(days=days), end, freq=f"{every}D"):
        r = requests.get("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi", timeout=120, params=dict(
            SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS=VNP_LAYER, CRS="EPSG:4326",
            BBOX=f"{la0},{lo0},{la1},{lo1}", FORMAT="image/png", TIME=str(d.date()),
            WIDTH=round((lo1 - lo0) / skyq.RES), HEIGHT=round((la1 - la0) / skyq.RES)))
        if r.ok:
            a = np.asarray(Image.open(io.BytesIO(r.content)).convert("LA"), float)
            stack.append(np.where(a[..., 1] > 0, a[..., 0], np.nan))
    g = np.nanmedian(np.array(stack), 0)
    os.makedirs(DATA, exist_ok=True)
    np.save(VNP_PATH, g)
    return g


def recent_emission():
    """VNP46A2 composite harmonised to the model's training scale (2016 Black Marble) by quantile mapping.
    ponytail: quantile mapping keeps the model's input distribution while updating *where* the lights are
    (Mosul was largely dark in 2016 under occupation). Retrain on VNP46A2 directly when Earthdata access is set up."""
    g = fetch_vnp46a2()
    old = skyclass.crop(skyclass.emission(skyclass.fetch_region()), skyclass.IRAQ_WORK)
    q = np.linspace(0, 100, 1001)
    return np.interp(np.nan_to_num(g, nan=np.nanmedian(g)), np.nanpercentile(g, q), np.percentile(old, q)).astype(np.float32)


# ---------------------------------------------------------------- static layers: elevation, roads
def elevation(lats, lons, coarse=0.08):
    """Open-Meteo elevation API (Copernicus 90 m DEM). Sampled on a ~9 km grid (few calls on the free API), each
    point takes the nearest sample. ponytail: plenty for a haze/altitude bonus; use full resolution if it matters."""
    import time
    lats, lons = np.asarray(lats), np.asarray(lons)
    ga = np.arange(lats.min(), lats.max() + coarse, coarse)
    go = np.arange(lons.min(), lons.max() + coarse, coarse)
    GO, GA = np.meshgrid(go, ga)
    pa, po, vals = GA.ravel(), GO.ravel(), []
    for i in range(0, len(pa), 100):
        for attempt in range(8):  # free API: back off on 429 / dropped connections
            try:
                r = requests.get("https://api.open-meteo.com/v1/elevation", timeout=60, params=dict(
                    latitude=",".join(f"{v:.3f}" for v in pa[i:i + 100]), longitude=",".join(f"{v:.3f}" for v in po[i:i + 100])))
                if r.status_code != 429:
                    r.raise_for_status()
                    break
            except requests.ConnectionError:
                pass
            time.sleep(5 * (attempt + 1))
        vals += r.json()["elevation"]
        time.sleep(2)
    grid = np.array(vals, float).reshape(GA.shape)
    iy = np.clip(np.rint((lats - ga[0]) / coarse).astype(int), 0, len(ga) - 1)
    ix = np.clip(np.rint((lons - go[0]) / coarse).astype(int), 0, len(go) - 1)
    return grid[iy, ix]


def road_distance_km(lats, lons, bbox=NINEVEH, classes="motorway|trunk|primary|secondary"):
    """Distance to the nearest major road from OpenStreetMap (Overpass; tries two public servers)."""
    import time
    la0, lo0, la1, lo1 = bbox
    q = f'[out:json][timeout:180];way["highway"~"^({classes})$"]({la0 - 0.3},{lo0 - 0.3},{la1 + 0.3},{lo1 + 0.3});out geom;'
    for attempt, url in enumerate(["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"] * 2):
        try:
            r = requests.post(url, data={"data": q}, timeout=240, headers={"User-Agent": "SkyQ-hackathon-prototype/0.1"})
            if r.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    res = 0.01  # rasterise roads on a ~1 km grid, then Euclidean distance transform
    gy0, gx0 = la1 + 0.3, lo0 - 0.3
    h, w = int((la1 - la0 + 0.6) / res) + 1, int((lo1 - lo0 + 0.6) / res) + 1
    grid = np.zeros((h, w), bool)
    for way in r.json()["elements"]:
        g = way.get("geometry", [])
        for a, b in zip(g, g[1:]):  # densify each segment so long straight roads are fully marked
            n = max(2, int(max(abs(a["lat"] - b["lat"]), abs(a["lon"] - b["lon"])) / (res / 2)))
            ys = ((gy0 - np.linspace(a["lat"], b["lat"], n)) / res).astype(int)
            xs = ((np.linspace(a["lon"], b["lon"], n) - gx0) / res).astype(int)
            ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            grid[ys[ok], xs[ok]] = True
    dist = distance_transform_edt(~grid, sampling=(res * 111.2, res * 111.2 * np.cos(np.radians((la0 + la1) / 2))))
    ys = ((gy0 - np.asarray(lats)) / res).astype(int).clip(0, h - 1)
    xs = ((np.asarray(lons) - gx0) / res).astype(int).clip(0, w - 1)
    return dist[ys, xs]


def cells(rebuild=False):
    """Static per-cell table for Nineveh: light-pollution score/class (AI + physics hybrid), night-light index,
    elevation, road distance."""
    if os.path.exists(CELLS_PATH) and not rebuild:
        return pd.read_csv(CELLS_PATH)
    model, _ = skyclass.load()
    em = recent_emission()
    grid = skyclass.predict_grid(model, em)
    # physics sky glow (Walker model calibrated on Globe at Night in Iraq), on the same recent lights
    phys = skyq.sqm_map(skyclass.crop(em, skyq.BBOX, skyclass.IRAQ_WORK))
    la = skyq.BBOX[2] - (np.arange(0, grid["cls"].shape[0], STEP) + 0.5) * skyq.RES
    lo = skyq.BBOX[1] + (np.arange(0, grid["cls"].shape[1], STEP) + 0.5) * skyq.RES
    LO, LA = np.meshgrid(lo, la)
    df = pd.DataFrame(dict(lat=LA.ravel(), lon=LO.ravel(), ai_score=grid["score"][::STEP, ::STEP].ravel(),
                           sqm=phys[::STEP, ::STEP].ravel(),
                           night_light=np.clip((grid["rad"][::STEP, ::STEP].ravel() + 4) / 4 * 100, 0, 100)))
    df = df[df.lat.between(NINEVEH[0], NINEVEH[2]) & df.lon.between(NINEVEH[1], NINEVEH[3])].reset_index(drop=True)
    # ponytail: hybrid = mean of AI score and physics score (SQM 16 -> 0 ... 22 -> 100). The AI alone under-rates
    # mid-size city cores (Macro-F1 0.38); physics alone can't learn from ground truth. Re-weight when Iraqi SQM data grows.
    df["physics_score"] = ((df.sqm - 16) / 6).clip(0, 1) * 100
    df["lp_score"] = (df.ai_score + df.physics_score) / 2
    df["lp_cls"] = to_class(df.lp_score)
    old = pd.read_csv(CELLS_PATH) if os.path.exists(CELLS_PATH) else None
    if old is not None and len(old) == len(df) and {"elevation_m", "road_km"} <= set(old.columns):
        df["elevation_m"], df["road_km"] = old.elevation_m.to_numpy(), old.road_km.to_numpy()  # static: don't refetch
    else:
        df["elevation_m"] = elevation(df.lat.to_numpy(), df.lon.to_numpy())
        df["road_km"] = road_distance_km(df.lat, df.lon)
    df.round(4).to_csv(CELLS_PATH, index=False)
    return df


def nearest(tab, lat, lon):
    return tab.iloc[int(((tab.lat - lat) ** 2 + (tab.lon - lon) ** 2).argmin())]


# ---------------------------------------------------------------- time-dependent conditions
def local_night(d: date):
    """Hourly local times for the night starting on d (18:00 -> 06:00)."""
    return [datetime.combine(d, datetime.min.time()) + timedelta(hours=18 + h) for h in range(13)]


def conditions(lats, lons, whens, tz="Asia/Baghdad"):
    """Cloud (total + low), humidity, visibility, dust/smoke (AOD, PM2.5) at local time(s) for one or many points.
    Rows are point-major: row = point_index * len(whens) + time_index. Falls back to climatology (flagged)."""
    whens = whens if isinstance(whens, list) else [whens]
    keys = [w.strftime("%Y-%m-%dT%H:00") for w in whens]
    n, notes = len(lats) * len(keys), []
    base = dict(latitude=",".join(f"{v:.3f}" for v in lats), longitude=",".join(f"{v:.3f}" for v in lons),
                timezone=tz, start_date=str(min(whens).date()), end_date=str(max(whens).date()))
    out = pd.DataFrame({k: np.full(n, v) for k, v in {**skyq.DEFAULT_WX, "cloud_cover_low": 0.0, "pm2_5": 10.0}.items()})

    def pull(url, fields, cols):
        try:
            r = requests.get(url, params=dict(base, hourly=",".join(fields)), timeout=60)
            r.raise_for_status()
            js = r.json()
            js = js if isinstance(js, list) else [js]
            for i, loc in enumerate(js):
                t = loc["hourly"]["time"]
                for j, key in enumerate(keys):
                    if key in t:
                        for f, c in zip(fields, cols):
                            v = loc["hourly"][f][t.index(key)]
                            if v is not None:
                                out.loc[i * len(keys) + j, c] = v / 1000 if f == "visibility" else v
            return True
        except Exception:
            return False

    if not pull("https://api.open-meteo.com/v1/forecast",
                ["cloud_cover", "cloud_cover_low", "relative_humidity_2m", "visibility", "wind_speed_10m", "wind_speed_250hPa"],
                ["cloud_cover", "cloud_cover_low", "humidity", "visibility_km", "wind10", "wind250"]):
        notes.append("weather")
    if not pull("https://air-quality-api.open-meteo.com/v1/air-quality", ["aerosol_optical_depth", "pm2_5"], ["aod", "pm2_5"]):
        notes.append("dust")
    return out, notes


def sky_geometry(lat, lon, times_local, tz="Asia/Baghdad"):
    """Sun altitude, Moon altitude and illumination at local wall-clock times (daylight saving handled by zoneinfo)."""
    utc = [t.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc).replace(tzinfo=None) for t in times_local]
    g = skyq.geometry(utc, lat, lon, None, radec=(0.0, 0.0, 0.0))
    return g[["sun_alt", "moon_alt", "moon_illum"]]


# ---------------------------------------------------------------- scoring
def effective_cloud(df, region="mosul"):
    """Cloud that actually blocks the sky. SF: sites >= 600 m usually sit above the marine layer, so most low cloud
    (fog) is below them. ponytail: fixed 70% discount; learn it from field reports."""
    if not REGIONS[region]["fog"]:
        return df.cloud_cover
    return (df.cloud_cover - np.where(df.elevation_m >= 600, 0.7, 0.0) * df.cloud_cover_low).clip(0, 100)


def components(df, region="mosul"):
    """0-100 component scores: light, weather, moon, elevation, smoke."""
    cloud = effective_cloud(df, region)
    weather = (100 * (1 - cloud / 100) ** 1.5 * (1 - ((df.humidity - 70) / 60).clip(0, 0.5))
               * (1 - (df.aod - 0.2)).clip(0.3, 1) * (df.visibility_km / 10).clip(0.3, 1))
    moon = np.where(df.moon_alt <= 0, 100.0,
                    100 * (1 - df.moon_illum ** 1.5 * np.sqrt(np.sin(np.radians(df.moon_alt.clip(0, 90))))))
    elev = (df.elevation_m / 1500).clip(0, 1) * 100
    smoke = 100 * (1 - (df.pm2_5 - 12) / 50).clip(0, 1)
    return pd.DataFrame(dict(light=df.lp_score.to_numpy(), weather=weather.to_numpy(), moon=np.clip(moon, 0, 100),
                             elevation=elev.to_numpy(), smoke=smoke.to_numpy()), index=df.index)


def sky_score(df, obs_type, region="mosul"):
    """Weighted Sky Score + class + hard caps."""
    R = REGIONS[region]
    c = components(df, region)
    total = c[["light", "weather", "moon", "elevation", "smoke"]].to_numpy() @ np.array(R["weights"][obs_type])
    total = np.where(df.sun_alt > -12, np.minimum(total, 10), total)  # not dark yet
    total = np.where(effective_cloud(df, region) >= 80, np.minimum(total, 20), total)  # overcast
    if R["fog"]:
        total = np.where((df.cloud_cover_low >= 70) & (df.elevation_m < 400), np.minimum(total, 25), total)  # in the fog
    if R["smoke"]:
        total = np.where(df.pm2_5 >= 55, np.minimum(total, 30), total)  # wildfire smoke
    return c.assign(score=np.round(total).astype(int), cls=to_class(total))


def labels(row):
    """Human labels for the result card."""
    lp = ["عالي 🔴", "متوسط 🟠", "واطي 🟢", "واطي جداً ⭐"][int(row.lp_cls)]
    moon = "واطي 🟢" if row.moon >= 80 else ("متوسط 🟠" if row.moon >= 50 else "عالي 🔴")
    return lp, moon


def suitable(row):
    """Which observation types this place/time supports (✓ / ✗)."""
    return {
        "stargazing": row.score >= 51,
        "milky_way": row.lp_cls >= 2 and row.moon >= 70 and row.weather >= 60,
        "astrophotography": row.lp_cls >= 2 and row.moon >= 70 and row.weather >= 70 and row.cloud_cover < 20,
        "meteors": row.lp_cls >= 1 and row.weather >= 60 and row.moon >= 50,
    }


# ---------------------------------------------------------------- public API
def _num(v, nd=1):
    return None if pd.isna(v) else round(float(v), nd)


def analyze(lat, lon, when: datetime, obs_type="stargazing", region="mosul"):
    """Everything the result card needs for one place and time."""
    R = REGIONS[region]
    cell = nearest(R["cells"](), lat, lon)
    cond, notes = conditions([lat], [lon], when, R["tz"])
    geo = sky_geometry(lat, lon, [when], R["tz"])
    df = pd.concat([cond, geo], axis=1).assign(lp_score=cell.lp_score, lp_cls=cell.lp_cls, elevation_m=cell.elevation_m)
    row = pd.concat([df, sky_score(df, obs_type, region)], axis=1).iloc[0]
    lp, moon = labels(row)
    best = best_time(lat, lon, when.date() if when.hour >= 12 else when.date() - timedelta(days=1), obs_type, region)
    return dict(lat=lat, lon=lon, when=when, type=obs_type, region=region, score=int(row.score), cls=int(row.cls),
                lp_cls=int(row.lp_cls), light_pollution=lp, moon_impact=moon, moon_score=round(float(row.moon)),
                night_light_index=round(float(cell.night_light)),
                cloud_cover=round(float(row.cloud_cover)), cloud_cover_low=round(float(row.cloud_cover_low)),
                humidity=round(float(row.humidity)), aod=round(float(row.aod), 2), pm2_5=round(float(row.pm2_5)),
                visibility_km=round(float(row.visibility_km), 1), wind10=round(float(row.wind10)),
                wind250=round(float(row.wind250)), seeing_arcsec=round(float(skyq.seeing(row.wind10, row.wind250)), 1),
                sun_alt=round(float(row.sun_alt), 1), moon_alt=round(float(row.moon_alt), 1),
                moon_illum=round(float(row.moon_illum), 2),
                elevation_m=round(float(cell.elevation_m)), road_km=_num(cell.road_km),
                sqm=_num(cell.sqm, 2) if "sqm" in cell else None,
                components={k: round(float(row[k])) for k in ("light", "weather", "moon", "elevation", "smoke")},
                suitable=suitable(row), recommended=bool(row.score >= 51), best_time=best["window"],
                hourly=best["hourly"], notes=notes)


def best_time(lat, lon, night: date, obs_type, region="mosul"):
    """Hourly Sky Score through the night at one place; best window = longest run >= 51 (Good)."""
    R = REGIONS[region]
    cell = nearest(R["cells"](), lat, lon)
    times = local_night(night)
    df, _ = conditions([lat], [lon], times, R["tz"])
    df = pd.concat([df, sky_geometry(lat, lon, times, R["tz"])], axis=1)
    df = df.assign(lp_score=cell.lp_score, lp_cls=cell.lp_cls, elevation_m=cell.elevation_m)
    s = sky_score(df, obs_type, region).assign(time=times)
    best, run = None, []
    for i, ok in enumerate(list(s.score >= 51) + [False]):
        if ok:
            run.append(i)
        elif run:
            if best is None or (len(run), s.score.iloc[run].mean()) > (len(best), s.score.iloc[best].mean()):
                best = run
            run = []
    window = None if best is None else (times[best[0]], times[best[-1]] + timedelta(hours=1))
    return dict(window=window, hourly=s[["time", "score", "cls"]])


def recommend(lat, lon, radius_km, when: datetime, obs_type="stargazing", n=3, max_road_km=5, min_sep_km=12, region="mosul"):
    """Rank reachable places (<= max_road_km from a major road) within radius_km by Sky Score at `when`.
    Light pollution pre-filters the 25 best candidates; weather is then fetched for all of them in one call."""
    R = REGIONS[region]
    tab = R["cells"]()
    d = skyq.haversine_km(lat, lon, tab.lat.to_numpy(), tab.lon.to_numpy())
    reachable = tab.road_km.isna() | (tab.road_km <= max_road_km)  # NaN = road data unavailable -> don't filter
    cand = tab.assign(km=d)[(d <= radius_km) & reachable].sort_values(["lp_score", "km"], ascending=[False, True])
    picked = []
    for r in cand.itertuples():
        if all(skyq.haversine_km(r.lat, r.lon, p.lat, p.lon) >= min_sep_km for p in picked):
            picked.append(r)
        if len(picked) == 25:
            break
    if not picked:
        return pd.DataFrame(), []
    c = pd.DataFrame(picked).drop(columns="Index").reset_index(drop=True)
    cond, notes = conditions(c.lat.tolist(), c.lon.tolist(), when, R["tz"])
    geo = sky_geometry(lat, lon, [when], R["tz"])  # Sun/Moon barely change across 100 km
    df = pd.concat([c, cond], axis=1).assign(**{k: float(geo[k].iloc[0]) for k in geo})
    df = pd.concat([df, sky_score(df, obs_type, region)], axis=1)
    return df.sort_values(["score", "km"], ascending=[False, True]).head(n).reset_index(drop=True), notes


if __name__ == "__main__":  # self-check
    t = cells(rebuild=True)
    print(t.describe().round(2))
    mosul = nearest(t, *skyq.MOSUL)
    hatra = nearest(t, 35.588, 42.718)
    print("Mosul cell", mosul.to_dict(), "\nHatra cell", hatra.to_dict())
    assert mosul.lp_score < 40, "central Mosul must not look dark with recent lights"
    now = datetime.combine(date.today(), datetime.min.time()) + timedelta(hours=22)
    a = analyze(*skyq.MOSUL, now, "milky_way")
    print({k: v for k, v in a.items() if k != "hourly"})
    assert a["cls"] <= 1 and not a["suitable"]["milky_way"]
    rec, notes = recommend(*skyq.MOSUL, 100, now, "milky_way")
    print(rec[["lat", "lon", "km", "road_km", "elevation_m", "lp_score", "cloud_cover", "score", "cls"]], notes)
    assert len(rec) and rec.score.iloc[0] > a["score"]
    noon = datetime.combine(date.today(), datetime.min.time()) + timedelta(hours=13)
    assert analyze(*skyq.MOSUL, noon)["score"] <= 10, "daytime gate"
    print("OK")
