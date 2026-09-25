"""SkyQ core: light-pollution map (NASA VIIRS Black Marble) + observation-quality model.

Pipeline
  1. Light pollution: VIIRS night-lights radiance -> Walker-law sky-glow convolution -> SQM (mag/arcsec^2) -> Bortle.
  2. Per-hour conditions: Open-Meteo weather + CAMS aerosols, astropy Sun/Moon/target geometry.
  3. ML: RandomForest predicts an observation-quality score (0..1); trained on physics-labelled
     synthetic nights + real field logs (observations.csv), with real logs weighted higher.
"""
import io
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(DATA, "model.joblib")
LOG_PATH = os.path.join(DATA, "observations.csv")

# ---------------------------------------------------------------- light pollution
BBOX = (29.0, 38.5, 37.5, 49.0)  # lat_min, lon_min, lat_max, lon_max (Iraq + neighbours' glow)
RES = 0.02  # deg/pixel (~2 km)
NINEVEH = (35.3, 41.2, 37.1, 44.0)  # study area: Mosul / Nineveh governorate (lat_min, lon_min, lat_max, lon_max)
MOSUL = (36.340, 43.130)


def in_nineveh(lat, lon):
    return NINEVEH[0] <= lat <= NINEVEH[2] and NINEVEH[1] <= lon <= NINEVEH[3]
NATURAL_SQM = 22.0  # natural zenith sky, V mag/arcsec^2
# Calibration knob, fitted by calibrate_glow_scale() on 45 Globe at Night field observations in the region
# (leave-one-out MAE 1.46 mag vs 1.82 for the earlier 2-anchor guess of 20). Independent check: 6 SQM-meter
# readings give ~zero bias at this scale. Refit when more Iraqi field readings exist.
GLOW_SCALE = 112.0
GLOW_RADIUS_KM = 200


def fetch_black_marble():
    """VIIRS Black Marble composite for the bbox from NASA GIBS (cached to disk). Returns luminance 0..255."""
    path = os.path.join(DATA, "blackmarble_iraq.png")
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        la0, lo0, la1, lo1 = BBOX
        r = requests.get(
            "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
            params=dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS="VIIRS_Black_Marble",
                        CRS="EPSG:4326", BBOX=f"{la0},{lo0},{la1},{lo1}", FORMAT="image/png", TIME="2016-01-01",
                        WIDTH=round((lo1 - lo0) / RES), HEIGHT=round((la1 - la0) / RES)),
            timeout=90)
        r.raise_for_status()
        Image.open(io.BytesIO(r.content)).save(path)
    return np.asarray(Image.open(path).convert("L"), float)


def emission_map(lum):
    """Black Marble PNG luminance -> relative upward radiance (background terrain texture removed, gamma undone)."""
    bg = np.percentile(lum, 70)
    return (np.clip(lum - bg, 0, None) / (255 - bg)) ** 2.2


def glow_kernel():
    """Walker's law: sky glow from a source falls as d^-2.5 (softened at 1 km)."""
    dy = RES * 111.2
    dx = dy * np.cos(np.radians((BBOX[0] + BBOX[2]) / 2))
    ny, nx = int(GLOW_RADIUS_KM / dy), int(GLOW_RADIUS_KM / dx)
    y, x = np.mgrid[-ny:ny + 1, -nx:nx + 1]
    d2 = (y * dy) ** 2 + (x * dx) ** 2
    k = (d2 + 1.0) ** -1.25
    k[d2 > GLOW_RADIUS_KM ** 2] = 0
    return k


def sqm_map(emission, scale=GLOW_SCALE):
    """Zenith sky brightness (mag/arcsec^2) for every pixel."""
    ratio = scale * np.clip(fftconvolve(emission, glow_kernel(), mode="same"), 0, None)
    return NATURAL_SQM - 2.5 * np.log10(1 + ratio)


def visible_stars(nelm):
    """Approx. stars above the horizon brighter than nelm (whole-sky counts: ~50 at mag 2, ~5000 at mag 6, halved)."""
    return 10 ** (0.5 * np.asarray(nelm) + 0.7) / 2


def colorize(sqm):
    """SQM grid -> RGBA PNG data URI for map overlays (bright city sky white/yellow, pristine sky transparent)."""
    import base64
    stops = np.array([15.5, 18.0, 19.5, 20.5, 21.3, 22.0])
    cols = np.array([[255, 255, 255, 230], [255, 220, 60, 220], [255, 120, 0, 200], [200, 0, 80, 170],
                     [60, 0, 140, 110], [0, 0, 40, 0]], float)
    rgba = np.stack([np.interp(sqm, stops, cols[:, i]) for i in range(4)], -1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def bortle(sqm):
    for lim, cls in [(21.99, 1), (21.89, 2), (21.69, 3), (20.49, 4), (19.50, 5), (18.94, 6), (18.38, 7), (17.80, 8)]:
        if sqm >= lim:
            return cls
    return 9


def pixel(lat, lon):
    return int((BBOX[2] - lat) / RES), int((lon - BBOX[1]) / RES)


def in_bbox(lat, lon):
    return BBOX[0] <= lat < BBOX[2] and BBOX[1] <= lon < BBOX[3]


def site_sqm(sqm, lat, lon):
    if not in_bbox(lat, lon):
        raise ValueError(f"Site {lat},{lon} is outside the mapped region {BBOX}")
    y, x = pixel(lat, lon)
    return float(sqm[y, x])


def intervention(emission, lat, lon, radius_km, shielded, dimming, warm_cct):
    """DarkSky 'responsible lighting' scenario applied to all lights within radius_km of a site.
    ponytail: first-order factors from DarkSky guidance; full shielding removes direct uplight (~half of
    sky glow from typical unshielded stock), dimming scales linearly, 3000K-or-warmer LEDs scatter ~20% less.
    Replace with measured before/after SQM if a municipality pilots it."""
    factor = (1 - 0.5 * shielded) * (1 - dimming) * (0.8 if warm_cct else 1.0)
    la = BBOX[2] - (np.arange(emission.shape[0]) + 0.5) * RES
    lo = BBOX[1] + (np.arange(emission.shape[1]) + 0.5) * RES
    LO, LA = np.meshgrid(lo, la)
    dist = haversine_km(lat, lon, LA, LO)
    out = emission.copy()
    out[dist <= radius_km] *= factor
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def darkest_sites(sqm, lat, lon, max_km, n=5, min_sep_km=25):
    """Best candidate observing sites within max_km of (lat, lon), spread at least min_sep_km apart."""
    la = BBOX[2] - (np.arange(sqm.shape[0]) + 0.5) * RES
    lo = BBOX[1] + (np.arange(sqm.shape[1]) + 0.5) * RES
    LO, LA = np.meshgrid(lo, la)
    dist = haversine_km(lat, lon, LA, LO)
    # ponytail: coarse 5-px stride keeps it fast; sites are ~10 km apart anyway.
    cand = [(sqm[i, j], LA[i, j], LO[i, j], dist[i, j])
            for i in range(0, sqm.shape[0], 5) for j in range(0, sqm.shape[1], 5) if dist[i, j] <= max_km]
    cand.sort(key=lambda c: (-round(c[0], 2), c[3]))  # darkest first, then closest
    picked = []
    for s, a, o, d in cand:
        if all(haversine_km(a, o, p["lat"], p["lon"]) >= min_sep_km for p in picked):
            picked.append(dict(lat=round(a, 3), lon=round(o, 3), sqm=round(s, 2), bortle=bortle(s), km_from_you=round(d)))
        if len(picked) == n:
            break
    return pd.DataFrame(picked)


# ---------------------------------------------------------------- astronomy
CATALOG = {  # name: (RA deg, Dec deg, V mag) ; None -> solar-system body via astropy
    "Sirius": (101.287, -16.716, -1.46), "Vega": (279.235, 38.784, 0.03), "Altair": (297.696, 8.868, 0.77),
    "Deneb": (310.358, 45.280, 1.25), "Betelgeuse": (88.793, 7.407, 0.5), "Polaris": (37.955, 89.264, 1.98),
    "Pleiades (M45)": (56.75, 24.117, 1.6), "Orion Nebula (M42)": (83.822, -5.391, 4.0),
    "Andromeda Galaxy (M31)": (10.685, 41.269, 3.4), "Hercules Cluster (M13)": (250.423, 36.461, 5.8),
    "Ring Nebula (M57)": (283.396, 33.029, 8.8), "Whirlpool Galaxy (M51)": (202.470, 47.195, 8.4),
    "Jupiter": None, "Saturn": None, "Mars": None,
}
PLANET_MAG = {"Jupiter": -2.3, "Saturn": 0.6, "Mars": 0.5}


def geometry(times_utc, lat, lon, target, radec=None):
    """Hourly Sun/Moon/target geometry via astropy. times_utc: list of naive UTC datetimes."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
    from astropy.time import Time

    t = Time(times_utc, scale="utc")
    frame = AltAz(obstime=t, location=EarthLocation(lat=lat * u.deg, lon=lon * u.deg))
    sun, moon = get_sun(t), get_body("moon", t)
    if radec is not None:
        tgt, mag = SkyCoord(radec[0] * u.deg, radec[1] * u.deg), radec[2]
    elif CATALOG[target] is None:
        tgt, mag = get_body(target.lower(), t), PLANET_MAG[target]
    else:
        ra, dec, mag = CATALOG[target]
        tgt = SkyCoord(ra * u.deg, dec * u.deg)
    sun_aa, moon_aa, tgt_aa = sun.transform_to(frame), moon.transform_to(frame), tgt.transform_to(frame)
    elong = sun.separation(moon).deg
    target_alt = tgt_aa.alt.deg
    return pd.DataFrame(dict(
        sun_alt=sun_aa.alt.deg,
        moon_alt=moon_aa.alt.deg,
        moon_illum=(1 - np.cos(np.radians(elong))) / 2,
        moon_sep=moon_aa.separation(tgt_aa).deg,
        target_alt=target_alt,
        target_az=tgt_aa.az.deg,
        airmass=airmass(target_alt),
        target_mag=np.full(len(t), mag, float),
    ))


def airmass(alt_deg):
    """Kasten & Young (1989); valid down to the horizon, capped at 40 below it."""
    alt = np.asarray(alt_deg, float)
    x = 1 / (np.sin(np.radians(np.clip(alt, 0, 90))) + 0.50572 * (np.clip(alt, 0, 90) + 6.07995) ** -1.6364)
    return np.where(alt > 0, x, 40.0)


# ---------------------------------------------------------------- weather
DEFAULT_WX = dict(cloud_cover=10.0, humidity=35.0, wind10=10.0, wind250=60.0, visibility_km=20.0, aod=0.3)


def weather(lat, lon, night: date, start_hour=18, hours=13):
    """Hourly weather + aerosols for night 18:00 -> next 06:00 local. Falls back to Iraqi autumn climatology
    (flagged) where the forecast horizon (16 d weather, 5 d aerosol) doesn't reach."""
    times, notes = None, []
    base = dict(latitude=lat, longitude=lon, timezone="auto", start_date=str(night), end_date=str(night + timedelta(days=(start_hour + hours - 1) // 24)))
    wx = aq = None
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params=dict(
            base, hourly="cloud_cover,relative_humidity_2m,wind_speed_10m,visibility,wind_speed_250hPa"))
        r.raise_for_status()
        wx = r.json()
    except Exception as e:
        notes.append(f"weather forecast unavailable ({type(e).__name__}); using climatology")
    try:
        r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=30,
                         params=dict(base, hourly="aerosol_optical_depth,dust"))
        r.raise_for_status()
        aq = r.json()
    except Exception as e:
        notes.append(f"aerosol forecast unavailable ({type(e).__name__}); using climatology AOD")

    start = datetime.combine(night, datetime.min.time()) + timedelta(hours=start_hour)
    local = [start + timedelta(hours=h) for h in range(hours)]
    offset = (wx or aq or {}).get("utc_offset_seconds")
    if offset is None:
        offset = round(lon / 15) * 3600  # ponytail: solar-time offset when both APIs are down
        notes.append("timezone guessed from longitude")
    df = pd.DataFrame(dict(local_time=local))
    df["utc"] = df.local_time - timedelta(seconds=offset)

    def pick(src, key, scale=1.0):
        if not src:
            return None
        s = pd.Series(src["hourly"][key], index=pd.to_datetime(src["hourly"]["time"]), dtype=float)
        v = s.reindex(df.local_time).to_numpy() * scale
        return None if np.isnan(v).all() else v

    cols = dict(cloud_cover=pick(wx, "cloud_cover"), humidity=pick(wx, "relative_humidity_2m"),
                wind10=pick(wx, "wind_speed_10m"), wind250=pick(wx, "wind_speed_250hPa"),
                visibility_km=pick(wx, "visibility", 1e-3), aod=pick(aq, "aerosol_optical_depth"))
    for k, v in cols.items():
        df[k] = DEFAULT_WX[k] if v is None else pd.Series(v).fillna(DEFAULT_WX[k]).to_numpy()
    if wx and cols["cloud_cover"] is None:
        notes.append("date beyond weather forecast horizon; using climatology")
    if aq and cols["aod"] is None:
        notes.append("date beyond aerosol forecast horizon (5 days); using climatology AOD 0.3")
    return df, notes


# ---------------------------------------------------------------- physics label + ML
FEATURES = ["cloud_cover", "humidity", "wind10", "wind250", "visibility_km", "aod", "site_sqm",
            "sun_alt", "moon_alt", "moon_illum", "moon_sep", "target_alt", "airmass", "target_mag", "aperture_mm"]


def seeing(wind10, wind250):
    """Seeing FWHM (arcsec). ponytail: empirical proxy - jet-stream (250 hPa) wind dominates turbulence.
    Swap for a DIMM or FWHM measured on the telescope's own frames when available."""
    return 0.7 + 0.018 * np.asarray(wind250) + 0.03 * np.asarray(wind10)


def sky_brightness(site_sqm, sun_alt, moon_alt, moon_illum, moon_sep, aod):
    """Effective zenith sky brightness (mag/arcsec^2) = light pollution + moonlight + twilight.
    ponytail: moon term is a compact fit to Krisciunas & Schaefer (full moon high ~ 18 mag/arcsec^2 at a dark site);
    aerosols brighten city glow (more scattering)."""
    base = 10 ** (-0.4 * (np.asarray(site_sqm) - NATURAL_SQM))  # in natural-sky units
    art = (base - 1) * np.clip(1 + 1.5 * (np.asarray(aod) - 0.3), 0.5, None)  # SQM calibration already has typical AOD
    up = np.clip(np.sin(np.radians(moon_alt)), 0, None)
    moon = 30 * np.asarray(moon_illum) ** 3 * np.sqrt(up) * (1 + 2 * np.exp(-np.asarray(moon_sep) / 25))
    twi = np.where(np.asarray(sun_alt) > -18, 10 ** ((np.asarray(sun_alt) + 18) / 3) - 1, 0)
    return NATURAL_SQM - 2.5 * np.log10(1 + np.clip(art, 0, None) + moon + twi)


def limiting_mag(sky, aperture_mm):
    """Naked-eye limit from sky brightness (Schaefer 1990 approx.), plus telescope light-grasp gain."""
    nelm = 7.93 - 5 * np.log10(10 ** (4.316 - np.asarray(sky) / 5) + 1)
    gain = np.where(np.asarray(aperture_mm) > 7, 5 * np.log10(np.clip(aperture_mm, 7, None) / 7) - 0.5, 0)
    return nelm + gain


K_TYPICAL = 0.13 + 1.086 * 0.3  # zenith extinction (mag) on an ordinary Iraqi night


def physics_score(df):
    """Physics-based quality 0..1: detection margin (limit mag - extincted target mag) x cloud/humidity/seeing
    penalties. Used to label synthetic training data and as a transparent reference."""
    d = {k: np.asarray(df[k], float) for k in FEATURES}
    sky = sky_brightness(d["site_sqm"], d["sun_alt"], d["moon_alt"], d["moon_illum"], d["moon_sep"], d["aod"])
    k_ext = 0.13 + 1.086 * d["aod"] + np.clip(10 - d["visibility_km"], 0, None) * 0.05  # Rayleigh+ozone+aerosol+haze
    # NELM is calibrated on Globe at Night eyes, which already see through a typical Iraqi zenith atmosphere
    # (AOD~0.3 -> k~0.46), so only extinction beyond that is charged.
    margin = limiting_mag(sky, d["aperture_mm"]) - (d["target_mag"] + k_ext * d["airmass"] - K_TYPICAL)
    s = 1 / (1 + np.exp(-(margin - 1.0) / 0.8))
    s *= (1 - d["cloud_cover"] / 100) ** 1.5
    s *= 1 - np.clip((d["humidity"] - 80) / 40, 0, 0.6)  # haze + dew on optics
    seeing_f = np.clip(1.25 - 0.18 * seeing(d["wind10"], d["wind250"]), 0.4, 1)
    s *= np.where(d["aperture_mm"] > 7, seeing_f, 1.0)  # turbulence blurs telescope detail, not naked-eye stars
    s *= np.clip((d["target_alt"] - 5) / 15, 0, 1)  # below ~5 deg: obstructed / hopeless
    return np.clip(s, 0, 1)


def synthetic(n, seed=0):
    """Random but physically plausible nights over Iraq, labelled by physics_score + noise."""
    rng = np.random.default_rng(seed)
    alt = rng.uniform(-10, 90, n)
    df = pd.DataFrame(dict(
        cloud_cover=np.where(rng.random(n) < 0.6, rng.uniform(0, 15, n), rng.uniform(0, 100, n)),
        humidity=rng.uniform(10, 100, n), wind10=rng.gamma(2, 6, n), wind250=rng.gamma(3, 25, n),
        visibility_km=rng.uniform(0.5, 50, n), aod=rng.gamma(2, 0.2, n), site_sqm=rng.uniform(15.5, 22, n),
        sun_alt=rng.uniform(-60, 0, n), moon_alt=rng.uniform(-60, 80, n), moon_illum=rng.random(n),
        moon_sep=rng.uniform(0, 180, n), target_alt=alt, airmass=airmass(alt),
        target_mag=rng.uniform(-2, 11, n), aperture_mm=rng.choice([7, 50, 80, 150, 200, 300, 400], n)))
    df["quality"] = np.clip(physics_score(df) + rng.normal(0, 0.03, n), 0, 1)
    return df


def load_logs():
    return pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame(columns=FEATURES + ["quality"])


def log_observation(row: dict):
    """Append a real field observation (features + observer's 0..1 quality rating)."""
    os.makedirs(DATA, exist_ok=True)
    df = pd.DataFrame([{k: row[k] for k in FEATURES + ["quality"]}])
    df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)


GOOD, FAIR = 0.6, 0.3  # score thresholds: >=0.6 GOOD (observe), 0.3-0.6 FAIR, <0.3 POOR
CLASSES = ["poor", "fair", "good"]


def to_class(score):
    return np.where(np.asarray(score) >= GOOD, "good", np.where(np.asarray(score) >= FAIR, "fair", "poor"))


def classification_metrics(y_true, y_pred, y_score=None):
    """Binary GOOD-vs-not metrics from continuous true/pred scores."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    t, p = np.asarray(y_true) >= GOOD, np.asarray(y_pred) >= GOOD
    out = dict(accuracy=accuracy_score(t, p), precision=precision_score(t, p, zero_division=0),
               recall=recall_score(t, p, zero_division=0), f1=f1_score(t, p, zero_division=0),
               confusion=confusion_matrix(t, p, labels=[False, True]).tolist())
    if y_score is not None and 0 < t.mean() < 1:
        out["roc_auc"] = roc_auc_score(t, y_score)
    return {k: (round(float(v), 3) if not isinstance(v, list) else v) for k, v in out.items()}


def baselines(df):
    """Rule-of-thumb decisions an observer uses today, scored as 0/1 'go' predictions."""
    dark = (df.sun_alt < -18) & (df.target_alt > 20)
    return {
        "Rule: clear sky (cloud<30%) + dark + target>20°": (dark & (df.cloud_cover < 30)).astype(float),
        "Rule: + no bright Moon (illum<50% or set)": (dark & (df.cloud_cover < 30) & ((df.moon_illum < 0.5) | (df.moon_alt < 0))).astype(float),
    }


def train(n=20000, real_weight=50, cv_folds=5):
    """RandomForest on synthetic + real logs. Returns (model, metrics) with a full evaluation report:
    regression (MAE/RMSE/R2), binary GOOD-vs-not (accuracy/precision/recall/F1/ROC-AUC/confusion),
    3-class poor/fair/good (accuracy, macro-F1, per-class report, confusion), k-fold CV, rule baselines."""
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                                 mean_absolute_error, mean_squared_error, r2_score)
    from sklearn.model_selection import KFold, train_test_split

    syn, real = synthetic(n), load_logs()
    tr, te = train_test_split(syn, test_size=0.2, random_state=0)
    X = pd.concat([tr[FEATURES], real[FEATURES]])
    y = pd.concat([tr.quality, real.quality])
    w = np.r_[np.ones(len(tr)), np.full(len(real), real_weight)]
    params = dict(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0)
    model = RandomForestRegressor(**params).fit(X, y, sample_weight=w)
    pred = np.clip(model.predict(te[FEATURES]), 0, 1)

    t3, p3 = to_class(te.quality), to_class(pred)
    rep = classification_report(t3, p3, labels=CLASSES, output_dict=True, zero_division=0)
    metrics = dict(
        trained=datetime.now().isoformat(timespec="seconds"), n_train=len(tr), n_test=len(te), real_logs=len(real),
        label_source="physics-generated synthetic nights + real field logs (x%d weight)" % real_weight,
        regression=dict(mae=round(mean_absolute_error(te.quality, pred), 3),
                        rmse=round(float(np.sqrt(mean_squared_error(te.quality, pred))), 3),
                        r2=round(r2_score(te.quality, pred), 3)),
        binary=classification_metrics(te.quality, pred, pred),
        three_class=dict(accuracy=round(accuracy_score(t3, p3), 3),
                         macro_f1=round(f1_score(t3, p3, labels=CLASSES, average="macro"), 3),
                         weighted_f1=round(f1_score(t3, p3, labels=CLASSES, average="weighted"), 3),
                         per_class={c: {k: round(v, 3) for k, v in rep[c].items()} for c in CLASSES},
                         confusion=confusion_matrix(t3, p3, labels=CLASSES).tolist()),
        class_balance={c: round(float(np.mean(t3 == c)), 3) for c in CLASSES},
    )

    # k-fold CV on a subsample (smaller forest keeps retraining in the app fast)
    sub = syn.sample(min(len(syn), 8000), random_state=1)
    f1s, maes = [], []
    for a, b in KFold(cv_folds, shuffle=True, random_state=0).split(sub):
        m = RandomForestRegressor(**{**params, "n_estimators": 60}).fit(sub.iloc[a][FEATURES], sub.iloc[a].quality)
        p = m.predict(sub.iloc[b][FEATURES])
        f1s.append(f1_score(sub.iloc[b].quality >= GOOD, p >= GOOD))
        maes.append(mean_absolute_error(sub.iloc[b].quality, p))
    metrics["cv"] = dict(folds=cv_folds, f1_mean=round(float(np.mean(f1s)), 3), f1_std=round(float(np.std(f1s)), 3),
                         mae_mean=round(float(np.mean(maes)), 3), mae_std=round(float(np.std(maes)), 3))

    metrics["baselines"] = {name: classification_metrics(te.quality, b) for name, b in baselines(te).items()}
    metrics["baselines"]["SkyQ Random Forest"] = metrics["binary"]

    if len(real) >= 5:  # agreement with real observer ratings (in-sample: logs are in training)
        metrics["real_logs_in_sample"] = classification_metrics(real.quality, model.predict(real[FEATURES]))

    os.makedirs(DATA, exist_ok=True)
    joblib.dump((model, metrics), MODEL_PATH)
    return model, metrics


# ---------------------------------------------------------------- light-pollution validation (Globe at Night)
GAN_PATH = os.path.join(DATA, "globe_at_night_region.csv")


def fetch_globe_at_night():
    """All Globe at Night yearly CSVs (2006-2025), filtered to BBOX, cached to disk."""
    import re
    if os.path.exists(GAN_PATH):
        return pd.read_csv(GAN_PATH, low_memory=False)
    html = requests.get("https://globeatnight.org/maps-data/", timeout=30).text
    parts = []
    for link in sorted(set(re.findall(r'href="(/documents/\d+/GaN\d{4}\.csv)"', html))):
        df = pd.read_csv(io.BytesIO(requests.get("https://globeatnight.org" + link, timeout=120).content),
                         low_memory=False, encoding_errors="replace")
        parts.append(df[df.Latitude.between(BBOX[0], BBOX[2]) & df.Longitude.between(BBOX[1], BBOX[3])])
    out = pd.concat(parts)
    os.makedirs(DATA, exist_ok=True)
    out.to_csv(GAN_PATH, index=False)
    return out


def gan_clean(gan):
    """Usable naked-eye records: valid limiting mag, <=1/4 cloud, de-duplicated."""
    g = gan[(gan.LimitingMag >= 0) & gan.CloudCover.isin(["clear", "1/4 of sky"])]
    return g.drop_duplicates(["Latitude", "Longitude", "UTDate", "LimitingMag"]).reset_index(drop=True)


def calibrate_glow_scale(emission, grid=np.geomspace(5, 2000, 80)):
    """Fit GLOW_SCALE to Globe at Night naked-eye limiting magnitudes (min MAE on a log grid).
    Leave-one-out: each point is predicted with a scale fitted on all the others -> honest out-of-sample error."""
    g = gan_clean(fetch_globe_at_night())
    conv = np.clip(fftconvolve(emission, glow_kernel(), mode="same"), 0, None)
    glow = np.array([conv[pixel(a, o)] for a, o in zip(g.Latitude, g.Longitude)])
    obs = g.LimitingMag.to_numpy(float)
    err = np.abs(np.array([limiting_mag(NATURAL_SQM - 2.5 * np.log10(1 + s * glow), 7) for s in grid]) - obs)
    n = len(obs)
    loo = [err[err[:, np.arange(n) != i].mean(1).argmin(), i] for i in range(n)]
    return dict(n=n, fitted_scale=round(float(grid[err.mean(1).argmin()]), 1), loo_mae=round(float(np.mean(loo)), 2),
                uncalibrated_scale=20.0, uncalibrated_mae=round(float(np.abs(
                    limiting_mag(NATURAL_SQM - 2.5 * np.log10(1 + 20.0 * glow), 7) - obs).mean()), 2))


def validate_light_pollution(emission, scale=GLOW_SCALE):
    """Satellite sky-glow model vs citizen-science field data.
    Globe at Night 'LimitingMag' is the faintest star-chart the observer matched (0-7, integer steps), compared
    with the model's naked-eye limiting magnitude at that pixel. Also compares SQM-meter readings directly."""
    from scipy.stats import spearmanr
    g = gan_clean(fetch_globe_at_night())
    sqm = sqm_map(emission, scale)
    g["model_sqm"] = [sqm[pixel(a, o)] for a, o in zip(g.Latitude, g.Longitude)]
    g["model_nelm"] = limiting_mag(g.model_sqm, 7)
    err = g.model_nelm - g.LimitingMag
    res = dict(n=len(g), mae=round(float(err.abs().mean()), 2), bias=round(float(err.mean()), 2),
               within_1mag=round(float((err.abs() <= 1).mean()), 3),
               spearman=round(float(spearmanr(g.model_nelm, g.LimitingMag).statistic), 3),
               # 'dark vs bright' decision: is the sky good enough to see the Milky Way (NELM >= 4)?
               **{f"dark_sky_{k}": v for k, v in classification_metrics(g.LimitingMag / 4 * GOOD, g.model_nelm / 4 * GOOD).items()})
    s = g[g.SQMReading.between(10, 23)]
    if len(s):
        res["sqm_meter"] = dict(n=len(s), mae=round(float((s.model_sqm - s.SQMReading).abs().mean()), 2),
                                bias=round(float((s.model_sqm - s.SQMReading).mean()), 2))
    g["Country"] = g.Country.fillna("unknown")
    return res, g[["UTDate", "Country", "Latitude", "Longitude", "CloudCover", "LimitingMag", "model_nelm", "SQMReading", "model_sqm"]]


def load_model():
    import joblib
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train()


def plan_night(model, lat, lon, night, target, site_sqm_value, aperture_mm, radec=None, threshold=0.6, wx=None):
    """Hourly forecast table + recommended window for one target and night (wx: pre-fetched weather() result)."""
    df, notes = (wx[0].copy(), wx[1]) if wx is not None else weather(lat, lon, night)
    df = pd.concat([df, geometry(list(df.utc.dt.to_pydatetime()), lat, lon, target, radec)], axis=1)
    df["site_sqm"] = site_sqm_value
    df["aperture_mm"] = aperture_mm
    df["seeing_arcsec"] = seeing(df.wind10, df.wind250)
    df["sky_sqm_now"] = sky_brightness(df.site_sqm, df.sun_alt, df.moon_alt, df.moon_illum, df.moon_sep, df.aod)
    df["score"] = np.clip(model.predict(df[FEATURES]), 0, 1)
    df["physics_score"] = physics_score(df)
    return df, recommend(df, threshold), notes


def recommend(df, threshold=0.6):
    """Longest run of consecutive hours with score >= threshold (ties -> higher mean score)."""
    best, run = None, []
    for i, ok in enumerate(list(df.score >= threshold) + [False]):
        if ok:
            run.append(i)
        elif run:
            key = (len(run), df.score.iloc[run].mean())
            if best is None or key > best[0]:
                best = (key, run)
            run = []
    if best is None:
        return None
    idx = best[1]
    return dict(start=df.local_time.iloc[idx[0]], end=df.local_time.iloc[idx[-1]] + timedelta(hours=1),
                mean_score=float(df.score.iloc[idx].mean()), night_mean=float(df.score.mean()))


def week_outlook(model, lat, lon, site_sqm_value, start: date, days=7):
    """Generic naked-eye stargazing quality for each night 18:00-06:00, one weather call for the whole week.
    ponytail: 'generic sky' = a star at the zenith 1.5 mag brighter than this site's best-case naked-eye limit, so the
    score reads 'how good is tonight compared with a clear, moonless night *here*'."""
    wx, notes = weather(lat, lon, start, start_hour=18, hours=24 * days - 12)
    wx = wx[(wx.local_time.dt.hour >= 18) | (wx.local_time.dt.hour <= 5)].reset_index(drop=True)
    ref_mag = float(limiting_mag(site_sqm_value, 7)) - 1.5
    g = geometry(list(wx.utc.dt.to_pydatetime()), lat, lon, None, radec=(0.0, 0.0, ref_mag))
    g["target_alt"], g["airmass"], g["moon_sep"] = 90.0, 1.0, 90 - g.moon_alt  # zenith
    df = pd.concat([wx, g], axis=1)
    df["site_sqm"], df["aperture_mm"] = site_sqm_value, 7
    df["score"] = np.clip(model.predict(df[FEATURES]), 0, 1)
    df["night"] = (df.local_time - timedelta(hours=12)).dt.date
    return df, notes


# ---------------------------------------------------------------- hilal (new crescent) visibility
HILAL = "🌙 Hilal – new crescent Moon"
ODEH_ZONES = [(5.65, "A", "Visible by naked eye"),
              (2.0, "B", "Visible with optical aid; may be seen by naked eye"),
              (-0.96, "C", "Visible with optical aid only"),
              (-np.inf, "D", "Not visible, even with a telescope")]
YALLOP_ZONES = [(0.216, "A"), (-0.014, "B"), (-0.160, "C"), (-0.232, "D"), (-0.293, "E"), (-np.inf, "F")]


def conjunctions(start_utc, days):
    """New-moon (geocentric conjunction) instants in [start, start+days], hourly search + linear interpolation."""
    import astropy.units as u
    from astropy.coordinates import GeocentricTrueEcliptic, get_body, get_sun
    from astropy.time import Time

    t = Time(start_utc) + np.arange(0, days * 24 + 1) * u.hour
    ecl = GeocentricTrueEcliptic()
    d = (get_body("moon", t).transform_to(ecl).lon.deg - get_sun(t).transform_to(ecl).lon.deg + 180) % 360 - 180
    out = []
    for i in np.where((d[:-1] < 0) & (d[1:] >= 0))[0]:  # moon overtakes sun (the +180/-180 wrap goes the other way)
        f = -d[i] / (d[i + 1] - d[i])
        out.append((t[i] + f * u.hour).to_datetime())
    return out


def odeh_zone(v):
    return next((z, txt) for lim, z, txt in ODEH_ZONES if v >= lim)


def hilal_evening(lat, lon, day: date, utc_offset_h=3.0):
    """Crescent visibility on the evening of `day` (local). Odeh (2004) criterion - fitted on 737 crescent
    observations - evaluated at Yallop's best time Tb = sunset + 4/9 lag, topocentric & airless."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
    from astropy.time import Time

    noon_utc = datetime.combine(day, datetime.min.time()) + timedelta(hours=12 - utc_offset_h)
    t = Time(noon_utc) + np.arange(0, 12 * 60) * u.min  # local 12:00 -> 24:00, 1-min steps
    frame = AltAz(obstime=t, location=EarthLocation(lat=lat * u.deg, lon=lon * u.deg))  # no pressure -> airless
    sun, moon = get_sun(t).transform_to(frame), get_body("moon", t).transform_to(frame)
    local = lambda i: noon_utc + timedelta(minutes=int(i), hours=utc_offset_h)
    h0 = -0.833  # upper limb on the horizon incl. standard refraction
    sets = lambda alt: next((i for i in range(1, len(alt)) if alt[i - 1] >= h0 > alt[i]), None)
    i_ss, i_ms = sets(sun.alt.deg), sets(moon.alt.deg)
    res = dict(date=day, sunset=local(i_ss) if i_ss else None, moonset=local(i_ms) if i_ms else None)

    prev = conjunctions(noon_utc - timedelta(days=30), 31)
    conj = max([c for c in prev if c <= noon_utc + timedelta(minutes=i_ss or 0)], default=None)
    res["conjunction_utc"] = conj
    res["age_h"] = (noon_utc + timedelta(minutes=i_ss) - conj).total_seconds() / 3600 if conj and i_ss else None
    if i_ss is None:
        return {**res, "zone": "D", "verdict": "No sunset found (polar/day-length edge case)"}
    if res["age_h"] is None or res["age_h"] > 26 * 24:  # last conjunction ~29 d ago -> next one is still ahead
        return {**res, "age_h": None, "zone": "D", "verdict": "Moon is before conjunction – impossible"}
    if res["age_h"] > 72:  # criterion is for the first evenings after conjunction only
        return {**res, "zone": "D", "verdict": f"Not a new-crescent evening (Moon {res['age_h'] / 24:.0f} days old)"}
    moon_up_at_sunset = moon.alt.deg[i_ss] > h0
    if not moon_up_at_sunset or (i_ms is not None and i_ms <= i_ss):
        return {**res, "lag_min": 0, "zone": "D", "verdict": "Moon sets before the Sun – impossible"}
    lag = (i_ms if i_ms is not None else len(t) - 1) - i_ss
    ib = i_ss + round(4 / 9 * lag)
    arcv = moon.alt.deg[ib] - sun.alt.deg[ib]
    daz = sun.az.deg[ib] - moon.az.deg[ib]
    arcl = moon[ib].separation(sun[ib]).deg
    sd = np.degrees(np.arcsin(1737.4 / moon.distance.km[ib])) * 60  # topocentric semi-diameter, arcmin
    w = sd * (1 - np.cos(np.radians(arcl)))  # crescent width, arcmin
    poly = -0.1018 * w ** 3 + 0.7319 * w ** 2 - 6.3226 * w
    v = arcv - (poly + 7.1651)
    q = (arcv - (poly + 11.8371)) / 10  # ponytail: Yallop defines ARCV geocentrically; topocentric here, secondary only
    zone, txt = odeh_zone(v)
    return {**res, "lag_min": lag, "best_time": local(ib), "moon_alt": float(moon.alt.deg[ib]),
            "moon_az": float(moon.az.deg[ib]), "arcv": float(arcv), "daz": float(daz), "arcl": float(arcl),
            "width_arcmin": float(w), "odeh_v": float(v), "zone": zone, "verdict": txt,
            "yallop_q": float(q), "yallop_zone": next(z for lim, z in YALLOP_ZONES if q >= lim)}


def hilal_weather_adjust(h, wx_row):
    """Downgrade the clear-sky Odeh zone for the actual atmosphere at best time (Iraqi dust matters most:
    the crescent sits a few degrees up, where airmass ~10 multiplies aerosol extinction).
    ponytail: 1 zone per 1.5 mag of extinction beyond Odeh's clear reference (AOD~0.1); cloud >=60% -> D, >=30% -> -1.
    Calibrate against logged Iraqi sighting reports."""
    if "odeh_v" not in h or wx_row is None:
        return {**h, "final_zone": h["zone"]}
    k = 0.13 + 1.086 * wx_row["aod"] + max(0, 10 - wx_row["visibility_km"]) * 0.05
    extra = max(0.0, (k - (0.13 + 1.086 * 0.1)) * float(airmass(max(h["moon_alt"], 0.5))))
    steps = int(extra // 1.5) + (1 if wx_row["cloud_cover"] >= 30 else 0) + (3 if wx_row["cloud_cover"] >= 60 else 0)
    zones = "ABCD"
    final = zones[min(3, zones.index(h["zone"]) + steps)]
    return {**h, "cloud_cover": wx_row["cloud_cover"], "aod": wx_row["aod"], "extra_extinction_mag": extra,
            "final_zone": final, "final_verdict": dict((z, t) for _, z, t in ODEH_ZONES)[final]}


def plan_hilal(lat, lon, day: date, nights=3):
    """Crescent forecast for `nights` evenings from `day`, clear-sky criterion + forecast weather at best time."""
    rows, notes = [], []
    for d in (day + timedelta(days=i) for i in range(nights)):
        wx, n = weather(lat, lon, d, start_hour=15, hours=7)
        notes += [x for x in n if x not in notes]
        off = (wx.local_time.iloc[0] - wx.utc.iloc[0]).total_seconds() / 3600
        h = hilal_evening(lat, lon, d, off)
        row = None
        if h.get("best_time"):
            row = wx.iloc[(wx.local_time - h["best_time"]).abs().argmin()].to_dict()
        rows.append(hilal_weather_adjust(h, row))
    return pd.DataFrame(rows), notes


# Clear-cut historical cases (crescent outcome reported worldwide). Sanity checks, not a statistical validation.
HILAL_CASES = [  # (evening, lat, lon, observed?, note)
    (date(2023, 3, 21), 36.34, 43.13, False, "Ramadan 1444 – before conjunction; not seen"),
    (date(2023, 3, 22), 36.34, 43.13, True, "Ramadan 1444 – seen across Middle East"),
    (date(2024, 4, 8), 36.34, 43.13, False, "Shawwal 1445 – before conjunction; not seen"),
    (date(2024, 4, 9), 36.34, 43.13, True, "Shawwal 1445 – seen; Eid 10 Apr"),
]


def validate_hilal():
    rows = []
    for d, la, lo, seen, note in HILAL_CASES:
        h = hilal_evening(la, lo, d)
        pred = h["zone"] in ("A", "B", "C")
        rows.append(dict(evening=d, note=note, age_h=round(h["age_h"], 1) if h.get("age_h") else None,
                         odeh_v=round(h["odeh_v"], 2) if "odeh_v" in h else None, zone=h["zone"],
                         predicted_visible=pred, observed_visible=seen, correct=pred == seen))
    return pd.DataFrame(rows)


if __name__ == "__main__":  # self-check: physics sanity + geometry + end-to-end
    base = dict(cloud_cover=0, humidity=30, wind10=5, wind250=40, visibility_km=40, aod=0.1, site_sqm=21.8,
                sun_alt=-40, moon_alt=-30, moon_illum=0.0, moon_sep=120, target_alt=60, airmass=1.15,
                target_mag=8.0, aperture_mm=150)
    s = lambda **kw: float(physics_score(pd.DataFrame([{**base, **kw}]))[0])
    assert s() > 0.8, s()
    assert s(site_sqm=17.5) < s(), "light pollution must hurt"
    assert s(cloud_cover=95) < 0.1, "overcast must be poor"
    assert s(moon_alt=60, moon_illum=1.0, moon_sep=20) < s(), "bright nearby moon must hurt"
    assert s(target_alt=2, airmass=float(airmass(2))) == 0, "below horizon"
    assert s(sun_alt=-5) < 0.1, "twilight"
    assert bortle(21.95) == 2 and bortle(17.0) == 9
    assert abs(float(airmass(90)) - 1.0) < 0.01
    sqm = sqm_map(emission_map(fetch_black_marble()))
    bgd, desert = site_sqm(sqm, 33.31, 44.37), site_sqm(sqm, 32.0, 41.0)
    print(f"SQM Baghdad={bgd:.2f} (Bortle {bortle(bgd)}), western desert={desert:.2f} (Bortle {bortle(desert)})")
    assert bgd < 18.5 and desert > 21.3
    g = geometry([datetime(2026, 9, 25, 18)], 33.31, 44.37, "Vega")
    assert 0 <= g.moon_illum[0] <= 1
    import json
    model, m = train()
    print("model metrics:", json.dumps(m, indent=1, default=str))
    assert m["regression"]["mae"] < 0.08 and m["binary"]["f1"] > 0.85
    assert m["binary"]["f1"] > max(b["f1"] for k, b in m["baselines"].items() if k.startswith("Rule")), "ML must beat rules"
    lp, pts = validate_light_pollution(emission_map(fetch_black_marble()))
    print("light pollution vs Globe at Night:", lp)
    df, rec, notes = plan_night(model, 33.31, 44.37, date.today(), "Vega", bgd, 150)
    print(df[["local_time", "cloud_cover", "aod", "target_alt", "moon_alt", "score"]].round({"aod": 2, "target_alt": 1, "moon_alt": 1, "score": 2}).to_string(index=False))
    print("recommend:", rec, notes)
    hv = validate_hilal()
    print(hv.to_string(index=False))
    assert hv.correct.all(), "hilal historical cases"
    nm = conjunctions(datetime(2026, 9, 25), 30)[0]
    print("next new moon (UTC):", nm)
    hd, n = plan_hilal(33.31, 44.37, nm.date(), 2)
    print(hd[["date", "age_h", "lag_min", "odeh_v", "zone", "final_zone"]].to_string(index=False), n)
    print("OK")
