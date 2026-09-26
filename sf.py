"""SkyQ San Francisco — prototype v2 (plans/sf_usa_3day_plan.json).

Same idea as Mosul, but where real ground truth exists:
  training   : Globe at Night, US West 2012-2021 (SQM first) + NASA Black Marble 2016 light features
  extra feats: elevation, distance to coast/large water, fraction of lit area within 10 km
  validation : spatial GroupKFold on 0.25 deg blocks, metrics on Bay Area points, South Bay hold-out
  physics    : Walker sky-glow recalibrated on Bay Area SQM readings (leave-one-out)
  hybrid     : AI + physics mix chosen by validation Macro-F1 (not fixed at 50/50)
  engine     : Bay Area cells (recent VNP46A2 lights, elevation, roads, coast) -> engine.py (region="sf")
Deviation from the plan, stated honestly: no Earthdata login here, so recent lights still come from the NASA GIBS
display images (the plan's own fallback), quantile-mapped to the 2016 training scale.
"""
import io
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter

import skyclass
import skyq

DATA = skyq.DATA
RES = skyq.RES
TRAIN = (30.0, -125.0, 45.0, -110.0)  # US West training box
STUDY = (36.4, -123.3, 38.6, -121.0)  # Bay Area (lat_min, lon_min, lat_max, lon_max)
WORK = (34.4, -125.3, 40.6, -119.0)   # STUDY + 2 deg of surrounding lights
SOUTH_BAY_LAT = 37.45                 # hold-out: Bay Area points south of this (San Jose -> Coe -> Pinnacles)
TZ = "America/Los_Angeles"
CENTER = (37.7749, -122.4194)
BLOCK_DEG = 0.25
FEATURES = skyclass.FEATURES + ["elevation_m", "coast_km", "lit_frac_10km"]
MODEL_PATH = os.path.join(DATA, "sf_skyclass.joblib")
TABLE_PATH = os.path.join(DATA, "sf_training_table.csv")
CELLS_PATH = os.path.join(DATA, "sf_cells.csv")
ELEV_CACHE = os.path.join(DATA, "sf_elev_cache.csv")
CONTRIB_PATH = os.path.join(DATA, "sf_ground_obs.csv")
FIELD_PATH = os.path.join(DATA, "sf_field_readings.csv")
NATURAL_SQM = 22.0
FETCH_ELEVATION = False  # True = fill the whole cache from Open-Meteo (slow on the free API)


def in_study(lat, lon):
    return STUDY[0] <= lat <= STUDY[2] and STUDY[1] <= lon <= STUDY[3]


# ---------------------------------------------------------------- satellite + land rasters (NASA GIBS)
def gibs(layer, bbox, when, path, mode="L", tile_deg=8.0):
    """GIBS WMS image for bbox, fetched in tiles, cached as PNG. Returns uint8 array (mode L) or alpha (mode A)."""
    if not os.path.exists(path):
        la0, lo0, la1, lo1 = bbox
        rows = []
        for a1 in np.arange(la1, la0, -tile_deg):
            a0 = max(la0, a1 - tile_deg)
            row = []
            for b0 in np.arange(lo0, lo1, tile_deg):
                b1 = min(lo1, b0 + tile_deg)
                p = dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS=layer, CRS="EPSG:4326",
                         BBOX=f"{a0},{b0},{a1},{b1}", FORMAT="image/png",
                         WIDTH=round((b1 - b0) / RES), HEIGHT=round((a1 - a0) / RES))
                if when:
                    p["TIME"] = when
                r = requests.get("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi", params=p, timeout=180)
                r.raise_for_status()
                im = Image.open(io.BytesIO(r.content)).convert("RGBA")
                row.append(np.asarray(im)[..., 3] if mode == "A" else np.asarray(im.convert("L")))
            rows.append(np.hstack(row))
        os.makedirs(DATA, exist_ok=True)
        Image.fromarray(np.vstack(rows).astype(np.uint8)).save(path)
    return np.asarray(Image.open(path))


def lights_2016(bbox, tag):
    return skyclass.emission(gibs("VIIRS_Black_Marble", bbox, "2016-01-01", os.path.join(DATA, f"bm2016_{tag}.png")).astype(np.float32))


def land(bbox, tag):
    return gibs("OSM_Land_Mask", bbox, None, os.path.join(DATA, f"land_{tag}.png"), mode="A") > 128


def recent_lights(days=48, every=6):
    """Median of recent VNP46A2 daily images over WORK, quantile-mapped to the 2016 training scale."""
    path = os.path.join(DATA, "sf_vnp46a2_work.npy")
    if not os.path.exists(path):
        end = date.today() - timedelta(days=2)
        stack = []
        for d in pd.date_range(end - timedelta(days=days), end, freq=f"{every}D"):
            try:
                g = gibs("VIIRS_SNPP_GapFilled_BRDF_Corrected_DayNightBand_Radiance", WORK, str(d.date()),
                         os.path.join(DATA, f"_tmp_vnp_{d.date()}.png")).astype(float)
                stack.append(np.where(g > 0, g, np.nan))
                os.remove(os.path.join(DATA, f"_tmp_vnp_{d.date()}.png"))
            except requests.RequestException:
                continue
        np.save(path, np.nanmedian(np.array(stack), 0))
    g = np.load(path)
    old = lights_2016(WORK, "sf_work")
    q = np.linspace(0, 100, 1001)
    return np.interp(np.nan_to_num(g, nan=np.nanmedian(g)), np.nanpercentile(g, q), np.percentile(old, q)).astype(np.float32)


def extra_stack(em, landmask, lat_top):
    """Plan's new SF features as rasters: distance to coast / large water (km), lit fraction within 10 km."""
    dy = RES * 111.2
    dx = dy * np.cos(np.radians(lat_top - em.shape[0] * RES / 2))
    coast = np.minimum(distance_transform_edt(landmask, sampling=(dy, dx)), 300).astype(np.float32)
    k = skyclass._disk(10, dy, dx)
    from scipy.signal import fftconvolve
    lit = np.clip(fftconvolve((em > 0.05).astype(np.float32), k, mode="same"), 0, 1).astype(np.float32)
    return {"coast_km": coast, "lit_frac_10km": lit}


# ---------------------------------------------------------------- elevation (point cache, 0.05 deg)
def elevation(lats, lons):
    """Open-Meteo elevation, rounded to 0.05 deg and cached on disk (free API: small batches, back-off)."""
    key = lambda a, o: (round(round(a / 0.05) * 0.05, 2), round(round(o / 0.05) * 0.05, 2))
    cache = pd.read_csv(ELEV_CACHE) if os.path.exists(ELEV_CACHE) else pd.DataFrame(columns=["lat", "lon", "elevation_m"])
    known = {(r.lat, r.lon): r.elevation_m for r in cache.itertuples()}
    keys = [key(a, o) for a, o in zip(lats, lons)]
    todo = sorted({k for k in keys if k not in known})
    if not FETCH_ELEVATION and len(known):
        # ponytail: prototype stops at the cached sample (~2k points); missing points take the nearest cached value
        ka = np.array(list(known))
        kv = np.array(list(known.values()), float)
        for k in todo:
            known[k] = kv[int(((ka[:, 0] - k[0]) ** 2 + (ka[:, 1] - k[1]) ** 2).argmin())]
        todo = []
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        for attempt in range(8):
            try:
                r = requests.get("https://api.open-meteo.com/v1/elevation", timeout=60, params=dict(
                    latitude=",".join(f"{a:.2f}" for a, _ in chunk), longitude=",".join(f"{o:.2f}" for _, o in chunk)))
                if r.status_code != 429:
                    r.raise_for_status()
                    break
            except requests.ConnectionError:
                pass
            time.sleep(5 * (attempt + 1))
        for k, v in zip(chunk, r.json()["elevation"]):
            known[k] = v
        pd.DataFrame([dict(lat=a, lon=o, elevation_m=v) for (a, o), v in known.items()]).to_csv(ELEV_CACHE, index=False)
        time.sleep(1.5)
    return np.array([known[k] for k in keys], float)


# ---------------------------------------------------------------- ground truth + training table
def ground_truth():
    g = skyclass.fetch_gan_all()
    la0, lo0, la1, lo1 = TRAIN
    g = g[g.Latitude.between(la0, la1 - RES) & g.Longitude.between(lo0, lo1 - RES)]
    g = g[g.CloudCover.isin(["clear", "1/4 of sky"])]
    year = pd.to_numeric(g.UTDate.astype(str).str[:4], errors="coerce")
    g = g[year.between(2012, 2021)].copy()
    has_sqm = g.SQMReading.between(14, 22.5)
    g = g[has_sqm | (g.LimitingMag >= 0)].copy()
    g["source"] = np.where(has_sqm[g.index], "SQM meter", "naked eye")
    g["label"] = np.where(g.source == "SQM meter", skyclass.label_sqm(g.SQMReading.fillna(0)), skyclass.label_nelm(g.LimitingMag))
    g = g.drop_duplicates(["Latitude", "Longitude", "UTDate", "label"])
    if os.path.exists(CONTRIB_PATH):
        c = pd.read_csv(CONTRIB_PATH)
        c = c[c.label >= 0].rename(columns={"lat": "Latitude", "lon": "Longitude", "date": "UTDate"}).assign(source="SkyQ contributor")
        g = pd.concat([g, c], ignore_index=True)
    return g.reset_index(drop=True)


def training_table(rebuild=False):
    if os.path.exists(TABLE_PATH) and not rebuild:
        return pd.read_csv(TABLE_PATH)
    g = ground_truth()
    em = lights_2016(TRAIN, "uswest")
    stack = skyclass.feature_stack(em, TRAIN[2])
    stack.update(extra_stack(em, land(TRAIN, "uswest"), TRAIN[2]))
    r = ((TRAIN[2] - g.Latitude) / RES).astype(int).to_numpy()
    c = ((g.Longitude - TRAIN[1]) / RES).astype(int).to_numpy()
    feats = pd.DataFrame({f: stack[f][r, c] for f in FEATURES if f != "elevation_m"})
    feats["elevation_m"] = elevation(g.Latitude.to_numpy(), g.Longitude.to_numpy())
    t = pd.concat([g[["Latitude", "Longitude", "UTDate", "source", "label", "SQMReading"]].reset_index(drop=True), feats], axis=1)
    t["block"] = (np.floor(t.Latitude / BLOCK_DEG) * 10000 + np.floor(t.Longitude / BLOCK_DEG)).astype(int)
    t["bay_area"] = [in_study(a, o) for a, o in zip(t.Latitude, t.Longitude)]
    t["south_bay"] = t.bay_area & (t.Latitude < SOUTH_BAY_LAT)
    t.to_csv(TABLE_PATH, index=False)
    return t


# ---------------------------------------------------------------- physics (Walker sky glow) calibrated on Bay Area SQM
def physics_sqm(log_glow, scale):
    return NATURAL_SQM - 2.5 * np.log10(1 + scale * 10 ** np.asarray(log_glow))


def calibrate_physics(t, grid=np.geomspace(1, 5000, 120)):
    s = t[t.bay_area & (t.source == "SQM meter")]
    obs = s.SQMReading.to_numpy()
    err = np.abs(np.array([physics_sqm(s.log_walker_glow, k) for k in grid]) - obs)
    n = len(obs)
    loo = [err[err[:, np.arange(n) != i].mean(1).argmin(), i] for i in range(n)]
    best = float(grid[err.mean(1).argmin()])
    bias = float((physics_sqm(s.log_walker_glow, best) - obs).mean())
    return dict(n=n, scale=round(best, 2), loo_mae=round(float(np.mean(loo)), 3), bias=round(bias, 3))


def physics_score(sqm):
    """SQM -> 0-100 so the class limits line up: SQM 18 / 19.5 / 21 land exactly on score 25 / 50 / 75."""
    return np.interp(np.asarray(sqm, float), [16.0, 18.0, 19.5, 21.0, 22.0], [0, 25, 50, 75, 100])


# ---------------------------------------------------------------- models
def _physics_rule_oof(t, groups, folds=5):
    from sklearn.model_selection import GroupKFold
    y, x = t.label.to_numpy(), t.log_walker_glow.to_numpy()
    oof = np.zeros(len(y), int)
    for tr, te in GroupKFold(folds).split(x, y, groups):
        cuts = [np.median([np.median(x[tr][y[tr] == k]), np.median(x[tr][y[tr] == k + 1])])
                if (y[tr] == k).any() and (y[tr] == k + 1).any() else np.inf for k in range(3)]
        oof[te] = 3 - np.digitize(x[te], sorted(cuts))
    return oof


def train_compare(folds=5, rebuild=False):
    """Compare LR / RF / XGBoost (spatial CV, Bay Area metrics, South Bay hold-out), SQM regression, physics
    calibration and the AI/physics mix. Saves the best model + full report."""
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBRegressor
    import engine

    t = training_table(rebuild)
    X, y, groups = t[FEATURES], t.label.to_numpy(), t.block.to_numpy()
    bay, south = t.bay_area.to_numpy(), t.south_bay.to_numpy()
    rnd = lambda d: {k: round(float(v), 3) for k, v in d.items()}
    rep = dict(trained=datetime.now().isoformat(timespec="seconds"), n=len(t), n_bay_area=int(bay.sum()),
               n_south_bay=int(south.sum()), n_blocks=int(len(np.unique(groups))), block_deg=BLOCK_DEG,
               class_counts_bay_area={c: int(((y == i) & bay).sum()) for i, c in enumerate(skyclass.CLASSES)},
               sources=t.source.value_counts().to_dict(), features=FEATURES, models={})
    best_name, oof_best, proba_best = None, None, None
    for name, m in skyclass.candidates().items():
        oof, proba = np.zeros(len(y), int), np.zeros((len(y), 4))
        for tr, te in GroupKFold(folds).split(X, y, groups):
            fitted = skyclass._fit(clone(m), X.iloc[tr], y[tr])
            proba[te] = fitted.predict_proba(X.iloc[te])
            oof[te] = proba[te].argmax(1)
        mi = skyclass._fit(clone(m), X[~south], y[~south])  # South Bay hold-out
        rep["models"][name] = dict(all_regions=rnd(skyclass._scores(y, oof)), bay_area=rnd(skyclass._scores(y[bay], oof[bay])),
                                   south_bay_holdout=rnd(skyclass._scores(y[south], mi.predict(X[south]))))
        if best_name is None or rep["models"][name]["bay_area"]["macro_f1"] > rep["models"][best_name]["bay_area"]["macro_f1"]:
            best_name, oof_best, proba_best = name, oof, proba
    rep["best"] = best_name
    phys_oof = _physics_rule_oof(t, groups, folds)
    maj = np.full(len(y), np.bincount(y).argmax())
    rep["baselines"] = {"Physics only (Walker glow thresholds)": dict(bay_area=rnd(skyclass._scores(y[bay], phys_oof[bay]))),
                        "Always majority class": dict(bay_area=rnd(skyclass._scores(y[bay], maj[bay])))}
    rep["per_class_bay_area"] = {k: {m: round(v, 3) for m, v in d.items()} for k, d in classification_report(
        y[bay], oof_best[bay], labels=range(4), target_names=skyclass.CLASSES, output_dict=True, zero_division=0).items()
        if k in skyclass.CLASSES}
    rep["confusion_bay_area"] = confusion_matrix(y[bay], oof_best[bay], labels=range(4)).tolist()

    # SQM regression (plan: report MAE in mag/arcsec²)
    s = t.source.to_numpy() == "SQM meter"
    ts = t[s]
    pred = np.zeros(s.sum())
    for tr, te in GroupKFold(folds).split(ts, groups=ts.block):
        pred[te] = XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, n_jobs=-1,
                                random_state=0).fit(ts[FEATURES].iloc[tr], ts.SQMReading.iloc[tr]).predict(ts[FEATURES].iloc[te])
    sb = ts.bay_area.to_numpy()
    rep["sqm_regression"] = dict(n=int(s.sum()), mae_all=round(float(mean_absolute_error(ts.SQMReading, pred)), 3),
                                 mae_bay_area=round(float(mean_absolute_error(ts.SQMReading[sb], pred[sb])), 3), n_bay_area=int(sb.sum()))

    # physics recalibration + AI/physics mix chosen by validation
    phys = calibrate_physics(t)
    rep["physics"] = phys
    ai_score = proba_best @ np.arange(4) / 3 * 100
    ph_score = physics_score(physics_sqm(t.log_walker_glow, phys["scale"]))
    mix = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        cls = engine.to_class(w * ai_score + (1 - w) * ph_score)
        mix[str(w)] = round(float(f1_score(y[bay], cls[bay], average="macro", labels=range(4))), 3)
    rep["hybrid_macro_f1_by_ai_weight"] = mix
    rep["hybrid_ai_weight"] = float(max(mix, key=mix.get))
    rep["hybrid_note"] = "physics scale fitted on the same Bay Area SQM points (slight optimism for the physics side)"

    model = skyclass._fit(skyclass.candidates()[best_name], X, y)
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    imp = getattr(last, "feature_importances_", None)
    imp = np.abs(last.coef_).mean(0) if imp is None else imp
    rep["feature_importance"] = dict(sorted(zip(FEATURES, map(lambda v: round(float(v), 4), imp)), key=lambda kv: -kv[1]))
    rep["field_test"] = evaluate_field(model, phys, rep["hybrid_ai_weight"])
    joblib.dump((model, rep), MODEL_PATH)
    return model, rep


def load():
    import joblib
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train_compare()


# ---------------------------------------------------------------- Bay Area cells for the decision engine
def cells(rebuild=False):
    """Per-cell table for the Bay Area (0.04 deg): AI score, physics SQM, hybrid light score, elevation, roads, coast."""
    if os.path.exists(CELLS_PATH) and not rebuild:
        return pd.read_csv(CELLS_PATH)
    import engine
    model, rep = load()
    em = recent_lights()
    stack = skyclass.feature_stack(em, WORK[2])
    stack.update(extra_stack(em, land(WORK, "sf_work"), WORK[2]))
    stack = {f: skyclass.crop(a, STUDY, WORK) for f, a in stack.items()}
    h, w = stack["log_rad"].shape
    la = STUDY[2] - (np.arange(h) + 0.5) * RES
    lo = STUDY[1] + (np.arange(w) + 0.5) * RES
    LO, LA = np.meshgrid(lo, la)
    step = 2
    sel = (slice(None, None, step), slice(None, None, step))
    df = pd.DataFrame(dict(lat=LA[sel].ravel(), lon=LO[sel].ravel()))
    df["elevation_m"] = elevation(df.lat.to_numpy(), df.lon.to_numpy())
    X = pd.DataFrame({f: stack[f][sel].ravel() for f in FEATURES if f != "elevation_m"}).assign(elevation_m=df.elevation_m)[FEATURES]
    P = model.predict_proba(X)
    ai = uniform_filter((P @ np.arange(4) / 3 * 100).reshape(LA[sel].shape), 3).ravel()  # ~12 km smoothing
    df["ai_score"] = ai
    df["sqm"] = physics_sqm(stack["log_walker_glow"][sel].ravel(), rep["physics"]["scale"])
    df["physics_score"] = physics_score(df.sqm)
    wgt = rep["hybrid_ai_weight"]
    df["lp_score"] = wgt * df.ai_score + (1 - wgt) * df.physics_score
    # class shown on the map = what was validated: the model's most likely class when the mix is AI-only
    df["lp_cls"] = P.argmax(1) if wgt == 1.0 else engine.to_class(df.lp_score)
    df["night_light"] = np.clip((stack["log_rad"][sel].ravel() + 4) / 4 * 100, 0, 100)
    df["coast_km"] = stack["coast_km"][sel].ravel()
    df["on_land"] = df.coast_km > 0
    try:
        df["road_km"] = engine.road_distance_km(df.lat, df.lon, bbox=STUDY, classes="motorway|trunk|primary")
    except Exception:  # Overpass is overloaded sometimes: keep the app working, mark reachability unknown
        df["road_km"] = np.nan
    df = df[df.on_land].reset_index(drop=True)
    df.round(4).to_csv(CELLS_PATH, index=False)
    return df


# ---------------------------------------------------------------- contributions + field test (the plan's final exam)
def contribute(lat, lon, milky_way: bool, sqm=None):
    label = int(skyclass.label_sqm(sqm)) if sqm else (2 if milky_way else 1)
    row = pd.DataFrame([dict(lat=lat, lon=lon, date=datetime.now().isoformat(timespec="minutes"), milky_way=milky_way,
                             SQMReading=sqm, label=label)])
    os.makedirs(DATA, exist_ok=True)
    row.to_csv(CONTRIB_PATH, mode="a", header=not os.path.exists(CONTRIB_PATH), index=False)
    return label


def evaluate_field(model=None, phys=None, ai_weight=0.5, path=FIELD_PATH):
    """Score the campaign's SQM readings (never used in training). Columns: lat, lon, sqm, [date, route, stop, notes].
    Returns None until the campaign has happened (planned around the 2026-10-10 new moon)."""
    if not os.path.exists(path):
        return None
    f = pd.read_csv(path).dropna(subset=["lat", "lon", "sqm"])
    if f.empty:
        return None
    em = lights_2016(WORK, "sf_work")
    stack = skyclass.feature_stack(em, WORK[2])
    stack.update(extra_stack(em, land(WORK, "sf_work"), WORK[2]))
    r = ((WORK[2] - f.lat) / RES).astype(int).to_numpy()
    c = ((f.lon - WORK[1]) / RES).astype(int).to_numpy()
    X = pd.DataFrame({k: stack[k][r, c] for k in FEATURES if k != "elevation_m"})
    X["elevation_m"] = elevation(f.lat.to_numpy(), f.lon.to_numpy())
    X = X[FEATURES]
    y = skyclass.label_sqm(f.sqm.to_numpy())
    import engine
    sqm_phys = physics_sqm(X.log_walker_glow, phys["scale"])
    hyb = ai_weight * (model.predict_proba(X) @ np.arange(4) / 3 * 100) + (1 - ai_weight) * physics_score(sqm_phys)
    out = {k: round(float(v), 3) for k, v in skyclass._scores(y, engine.to_class(hyb)).items()}
    out.update(n=len(f), physics_sqm_mae=round(float(np.abs(sqm_phys - f.sqm).mean()), 3))
    return out


if __name__ == "__main__":  # builds everything, prints the report, sanity checks
    import json
    if not os.path.exists(FIELD_PATH):  # template for the campaign team
        pd.DataFrame(columns=["date", "route", "stop", "lat", "lon", "sqm", "cloud", "fog", "moon_alt", "milky_way", "notes"]).to_csv(FIELD_PATH, index=False)
    m, rep = train_compare(rebuild=True)
    print(json.dumps({k: v for k, v in rep.items() if k != "features"}, indent=1, ensure_ascii=False))
    c = cells(rebuild=True)
    near = lambda a, o: c.iloc[int(((c.lat - a) ** 2 + (c.lon - o) ** 2).argmin())]
    sfc, pin = near(*CENTER), near(36.494, -121.146)
    print("SF downtown", sfc[["ai_score", "sqm", "lp_score", "lp_cls", "elevation_m", "road_km"]].to_dict())
    print("Pinnacles  ", pin[["ai_score", "sqm", "lp_score", "lp_cls", "elevation_m", "road_km"]].to_dict())
    assert sfc.lp_score < pin.lp_score, "downtown SF must score worse than Pinnacles"
    print("OK")
