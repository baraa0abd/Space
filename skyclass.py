"""Sky Quality classifier: NASA VIIRS night lights -> ground sky quality (Poor / Moderate / Good / Excellent).

Satellites measure light going *up*; observers care how dark the sky looks from the *ground*. This module learns that
mapping from real ground truth - Globe at Night citizen measurements (naked-eye limiting magnitude + SQM meter
readings) across Europe, the Middle East and North Africa - using light-environment features computed from the
VIIRS Black Marble composite. Three models are compared (Logistic Regression, Random Forest, XGBoost) with
*spatial* GroupKFold (2x2 degree blocks) and a fully held-out Iraq test; the best Macro-F1 wins.
"""
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.signal import fftconvolve

import skyq

DATA = skyq.DATA
REGION = (25.0, -10.0, 48.0, 60.0)  # lat_min, lon_min, lat_max, lon_max: Mediterranean + Middle East + Gulf + Iran
RES = skyq.RES  # 0.02 deg, same grid as the Iraq map
CLASSES = ["Poor", "Moderate", "Good", "Excellent"]
CLASSES_AR = ["ضعيفة", "متوسطة", "جيدة", "ممتازة"]
EMOJI = ["🔴", "🟠", "🟢", "⭐"]
COLORS = [[220, 40, 40], [245, 150, 20], [40, 180, 80], [70, 110, 255]]
BG_LUM = 40.0  # ponytail: Black Marble base-map darkness level (land ~36-40); fixed so every crop gets the same emission
IRAQ_PAD = 2.0  # deg of surrounding lights included when computing Iraq features (> 100 km neighbourhood)
FEATURES = ["log_rad", "log_mean_2km", "log_mean_5km", "log_mean_10km", "log_mean_25km", "log_mean_50km",
            "log_mean_100km", "log_max_10km", "log_max_25km", "log_walker_glow", "dist_bright_km"]
SQM_EDGES = [18.0, 19.5, 21.0]  # Poor <18 (Bortle 8-9) | Moderate <19.5 (6-7) | Good <21 (4-5) | Excellent (1-3)
NELM_EDGES = [4, 5, 6]  # same classes in naked-eye limiting magnitude (Schaefer: SQM 18/19.5/21 ~ NELM 4.0/4.9/5.9)
MODEL_PATH = os.path.join(DATA, "skyclass.joblib")
TABLE_PATH = os.path.join(DATA, "skyclass_training.csv")
CONTRIB_PATH = os.path.join(DATA, "ground_obs.csv")
GAN_ALL = os.path.join(DATA, "gan_all.csv")


# ---------------------------------------------------------------- satellite raster
def fetch_region():
    """Black Marble luminance for REGION, fetched from NASA GIBS in tiles and cached."""
    path = os.path.join(DATA, "blackmarble_region.png")
    if not os.path.exists(path):
        la0, lo0, la1, lo1 = REGION
        rows = []
        for a0, a1 in [(la1 - (la1 - la0) / 2, la1), (la0, la0 + (la1 - la0) / 2)]:  # north strip first
            row = []
            for b0, b1 in [(lo0, lo0 + (lo1 - lo0) / 2), (lo0 + (lo1 - lo0) / 2, lo1)]:
                r = requests.get("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi", timeout=180, params=dict(
                    SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS="VIIRS_Black_Marble", CRS="EPSG:4326",
                    BBOX=f"{a0},{b0},{a1},{b1}", FORMAT="image/png", TIME="2016-01-01",
                    WIDTH=round((b1 - b0) / RES), HEIGHT=round((a1 - a0) / RES)))
                r.raise_for_status()
                row.append(np.asarray(Image.open(io.BytesIO(r.content)).convert("L")))
            rows.append(np.hstack(row))
        os.makedirs(DATA, exist_ok=True)
        Image.fromarray(np.vstack(rows)).save(path)
    return np.asarray(Image.open(path), np.float32)


def emission(lum):
    e = (np.clip(lum - BG_LUM, 0, None) / (255 - BG_LUM)) ** 2.2
    return np.where(e < 0.003, 0, e).astype(np.float32)  # noise floor: base-map texture is not light


def crop(arr, bbox, origin=REGION):
    """Sub-array of a REGION-aligned raster for bbox (lat_min, lon_min, lat_max, lon_max)."""
    r0, c0 = round((origin[2] - bbox[2]) / RES), round((bbox[1] - origin[1]) / RES)
    return arr[r0:r0 + round((bbox[2] - bbox[0]) / RES), c0:c0 + round((bbox[3] - bbox[1]) / RES)]


# ---------------------------------------------------------------- features
def _disk(r_km, dy, dx, power=None):
    ny, nx = max(1, int(r_km / dy)), max(1, int(r_km / dx))
    y, x = np.mgrid[-ny:ny + 1, -nx:nx + 1]
    d2 = (y * dy) ** 2 + (x * dx) ** 2
    if power is None:  # flat disk, normalised -> neighbourhood mean
        k = (d2 <= r_km ** 2).astype(np.float32)
        return k / k.sum()
    k = ((d2 + 1.0) ** power).astype(np.float32)  # Walker d^-2.5 sky-glow kernel
    k[d2 > r_km ** 2] = 0
    return k


def feature_stack(em, lat_top, band_deg=5.0, pad_deg=2.0):
    """Per-pixel features (dict name -> 2-D array). Processed in latitude bands so km-based kernels stay true
    (a degree of longitude shrinks with latitude)."""
    h, w = em.shape
    out = {f: np.zeros((h, w), np.float32) for f in FEATURES}
    step, pad = int(band_deg / RES), int(pad_deg / RES)
    dy = RES * 111.2
    for r0 in range(0, h, step):
        r1 = min(h, r0 + step)
        p0, p1 = max(0, r0 - pad), min(h, r1 + pad)
        band = em[p0:p1]
        dx = dy * np.cos(np.radians(lat_top - (r0 + r1) / 2 * RES))
        sl = slice(r0 - p0, r0 - p0 + (r1 - r0))
        conv = lambda k: np.clip(fftconvolve(band, k, mode="same"), 0, None)[sl]
        out["log_rad"][r0:r1] = np.log10(band[sl] + 1e-4)
        for r in (2, 5, 10, 25, 50, 100):
            out[f"log_mean_{r}km"][r0:r1] = np.log10(conv(_disk(r, dy, dx)) + 1e-4)
        for r in (10, 25):
            size = (2 * max(1, int(r / dy)) + 1, 2 * max(1, int(r / dx)) + 1)
            out[f"log_max_{r}km"][r0:r1] = np.log10(maximum_filter(band, size=size)[sl] + 1e-4)
        out["log_walker_glow"][r0:r1] = np.log10(conv(_disk(200, dy, dx, power=-1.25)) + 1e-4)
        bright = band > 0.05  # ponytail: "lit settlement" threshold on relative radiance
        dist = distance_transform_edt(~bright, sampling=(dy, dx)) if bright.any() else np.full(band.shape, 250.0)
        out["dist_bright_km"][r0:r1] = np.minimum(dist[sl], 250)
    return out


def sample(stack, lats, lons, lat_top, lon_left):
    r = ((lat_top - np.asarray(lats)) / RES).astype(int)
    c = ((np.asarray(lons) - lon_left) / RES).astype(int)
    return pd.DataFrame({f: stack[f][r, c] for f in FEATURES})


# ---------------------------------------------------------------- labels (ground truth)
def label_sqm(sqm):
    return np.digitize(sqm, SQM_EDGES)


def label_nelm(nelm):
    return np.digitize(nelm, NELM_EDGES)


def fetch_gan_all():
    """All Globe at Night yearly CSVs (2006-2025), essential columns only, cached."""
    import re
    if not os.path.exists(GAN_ALL):
        html = requests.get("https://globeatnight.org/maps-data/", timeout=30).text
        cols = ["Latitude", "Longitude", "UTDate", "LimitingMag", "SQMReading", "CloudCover", "Country"]
        parts = [pd.read_csv(io.BytesIO(requests.get("https://globeatnight.org" + l, timeout=180).content),
                             low_memory=False, encoding_errors="replace")[cols]
                 for l in sorted(set(re.findall(r'href="(/documents/\d+/GaN\d{4}\.csv)"', html)))]
        os.makedirs(DATA, exist_ok=True)
        pd.concat(parts).to_csv(GAN_ALL, index=False)
    return pd.read_csv(GAN_ALL, low_memory=False)


def ground_truth():
    """Labelled ground observations in REGION: SQM meter label when available (more precise), else naked-eye.
    Clear or <=1/4 cloud, VIIRS era 2012-2021 (Black Marble composite is 2016), de-duplicated."""
    g = fetch_gan_all()
    la0, lo0, la1, lo1 = REGION
    g = g[g.Latitude.between(la0, la1 - RES) & g.Longitude.between(lo0, lo1 - RES)]
    g = g[g.CloudCover.isin(["clear", "1/4 of sky"])]
    year = pd.to_numeric(g.UTDate.astype(str).str[:4], errors="coerce")
    g = g[year.between(2012, 2021)].copy()
    has_sqm = g.SQMReading.between(14, 22.5)
    g = g[has_sqm | (g.LimitingMag >= 0)].copy()
    g["source"] = np.where(has_sqm[g.index], "SQM meter", "naked eye")
    g["label"] = np.where(g.source == "SQM meter", label_sqm(g.SQMReading.fillna(0)), label_nelm(g.LimitingMag))
    g = g.drop_duplicates(["Latitude", "Longitude", "UTDate", "label"])
    if os.path.exists(CONTRIB_PATH):  # app contributions from Iraqi observers
        c = pd.read_csv(CONTRIB_PATH)
        c = c[c.label >= 0].assign(source="SkyQ contributor", Country="Iraq")
        g = pd.concat([g, c.rename(columns={"lat": "Latitude", "lon": "Longitude", "date": "UTDate"})], ignore_index=True)
    return g.reset_index(drop=True)


def contribute(lat, lon, milky_way: bool, sqm=None):
    """Store an observation from the app. Label: SQM reading if given; else Milky Way visible -> Good,
    not visible -> Moderate (ponytail: coarse, only 'at least'/'at most' information - SQM readings are better)."""
    label = int(label_sqm(sqm)) if sqm else (2 if milky_way else 1)
    row = pd.DataFrame([dict(lat=lat, lon=lon, date=datetime.now().isoformat(timespec="minutes"),
                             milky_way=milky_way, SQMReading=sqm, label=label)])
    os.makedirs(DATA, exist_ok=True)
    row.to_csv(CONTRIB_PATH, mode="a", header=not os.path.exists(CONTRIB_PATH), index=False)
    return label


def training_table(rebuild=False):
    """Ground truth joined with satellite features at each observation's pixel."""
    if os.path.exists(TABLE_PATH) and not rebuild:
        return pd.read_csv(TABLE_PATH)
    g = ground_truth()
    stack = feature_stack(emission(fetch_region()), REGION[2])
    t = pd.concat([g[["Latitude", "Longitude", "UTDate", "Country", "source", "label", "SQMReading", "LimitingMag"]],
                   sample(stack, g.Latitude, g.Longitude, REGION[2], REGION[1])], axis=1)
    t["block"] = (np.floor(t.Latitude / 2) * 1000 + np.floor(t.Longitude / 2)).astype(int)  # 2x2 deg spatial groups
    t.to_csv(TABLE_PATH, index=False)
    return t


# ---------------------------------------------------------------- models
def candidates():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    return {
        "Logistic Regression (baseline)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=0),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=-1, random_state=0, eval_metric="mlogloss"),
    }


def _fit(model, X, y):
    from sklearn.utils.class_weight import compute_sample_weight
    w = compute_sample_weight("balanced", y)
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    if type(last).__name__ == "XGBClassifier":  # xgboost has no class_weight -> balanced sample weights
        return model.fit(X, y, **({"xgbclassifier__sample_weight": w} if hasattr(model, "steps") else {"sample_weight": w}))
    return model.fit(X, y)


def _scores(y, p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
    return dict(macro_f1=f1_score(y, p, average="macro", labels=range(4), zero_division=0),
                accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                weighted_f1=f1_score(y, p, average="weighted", labels=range(4), zero_division=0),
                within_one_class=float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) <= 1)),
                quadratic_kappa=cohen_kappa_score(y, p, weights="quadratic"))


def train_compare(folds=5, rebuild=False):
    """Spatial GroupKFold comparison of 3 models, pick best Macro-F1, then: Iraq hold-out test, per-class report,
    confusion matrix, feature importances; final model refit on all data. Returns (model, report)."""
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupKFold

    t = training_table(rebuild)
    X, y, groups = t[FEATURES], t.label.to_numpy(), t.block.to_numpy()
    iraq = t.Latitude.between(skyq.BBOX[0], skyq.BBOX[2]) & t.Longitude.between(skyq.BBOX[1], skyq.BBOX[3])
    report = dict(trained=datetime.now().isoformat(timespec="seconds"), n=len(t), n_blocks=int(len(np.unique(groups))),
                  classes=CLASSES, class_counts={c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
                  sources=t.source.value_counts().to_dict(), models={})
    oof_best = None
    for name, m in candidates().items():
        oof = np.zeros(len(y), int)
        fold_f1 = []
        for tr, te in GroupKFold(folds).split(X, y, groups):
            oof[te] = _fit(clone(m), X.iloc[tr], y[tr]).predict(X.iloc[te])
            fold_f1.append(_scores(y[te], oof[te])["macro_f1"])
        s = {k: round(float(v), 3) for k, v in _scores(y, oof).items()}
        s["macro_f1_fold_std"] = round(float(np.std(fold_f1)), 3)
        # Iraq hold-out: train on everything outside Iraq's map box, test on the Iraqi-region points
        if iraq.sum() >= 5:
            mi = _fit(clone(m), X[~iraq], y[~iraq])
            s["iraq_holdout"] = {k: round(float(v), 3) for k, v in _scores(y[iraq], mi.predict(X[iraq])).items()}
        report["models"][name] = s
        if oof_best is None or s["macro_f1"] > report["models"][report["best"]]["macro_f1"]:
            report["best"], oof_best = name, oof
    # dumb baselines for context
    report["baselines"] = {
        "Always majority class": {k: round(float(v), 3) for k, v in _scores(y, np.full(len(y), np.bincount(y).argmax())).items()},
        "Physics only (Walker glow thresholds)": _physics_baseline(t, groups, folds),
    }
    best = report["best"]
    report["per_class"] = {k: {m: round(v, 3) for m, v in d.items()} for k, d in classification_report(
        y, oof_best, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0).items() if k in CLASSES}
    report["confusion"] = confusion_matrix(y, oof_best, labels=range(4)).tolist()
    report["iraq_points"] = int(iraq.sum())
    model = _fit(candidates()[best], X, y)
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    imp = getattr(last, "feature_importances_", None)
    if imp is None:
        imp = np.abs(last.coef_).mean(0)
    report["feature_importance"] = dict(sorted(zip(FEATURES, map(lambda v: round(float(v), 4), imp)), key=lambda kv: -kv[1]))
    joblib.dump((model, report), MODEL_PATH)
    return model, report


def _physics_baseline(t, groups, folds):
    """1-feature rule: class = thresholds on Walker sky glow, thresholds learned per fold (no ML)."""
    from sklearn.model_selection import GroupKFold
    y, x = t.label.to_numpy(), t.log_walker_glow.to_numpy()
    oof = np.zeros(len(y), int)
    for tr, te in GroupKFold(folds).split(x, y, groups):
        cuts = [np.median([np.percentile(x[tr][y[tr] == k], 50), np.percentile(x[tr][y[tr] == k + 1], 50)])
                if (y[tr] == k).any() and (y[tr] == k + 1).any() else np.inf for k in range(3)]
        oof[te] = 3 - np.digitize(x[te], sorted(cuts))  # more glow -> worse class
    return {k: round(float(v), 3) for k, v in _scores(y, oof).items()}


def load():
    import joblib
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train_compare()


# ---------------------------------------------------------------- inference for Iraq
IRAQ_WORK = (skyq.BBOX[0] - IRAQ_PAD, skyq.BBOX[1] - IRAQ_PAD, skyq.BBOX[2] + IRAQ_PAD, skyq.BBOX[3] + IRAQ_PAD)


def iraq_emission():
    return crop(emission(fetch_region()), IRAQ_WORK)


def predict_grid(model, em_work):
    """Class probabilities, class and 0-100 score for every pixel of the Iraq map box.
    Score = expected class on a 0-100 scale (Poor 0, Moderate 33, Good 67, Excellent 100)."""
    stack = feature_stack(em_work, IRAQ_WORK[2])
    stack = {f: crop(a, skyq.BBOX, IRAQ_WORK) for f, a in stack.items()}
    h, w = stack[FEATURES[0]].shape
    X = pd.DataFrame({f: stack[f].ravel() for f in FEATURES})
    P = model.predict_proba(X)
    # ponytail: 5-px (~10 km) smoothing - real sky glow varies smoothly; removes pixel-level classifier speckle
    from scipy.ndimage import uniform_filter
    score = uniform_filter((P @ np.array([0, 1, 2, 3]) / 3 * 100).reshape(h, w), 5)
    cls = np.digitize(score, [25, 50, 75])  # class from the (smoothed) expected score
    return dict(proba=P.reshape(h, w, 4), cls=cls, score=score,
                glow=stack["log_walker_glow"], rad=stack["log_rad"])


def at(grid, lat, lon):
    y, x = skyq.pixel(lat, lon)
    return dict(score=float(grid["score"][y, x]), cls=int(grid["cls"][y, x]), proba=grid["proba"][y, x].tolist(),
                night_light_index=float(np.clip((grid["rad"][y, x] + 4) / 4 * 100, 0, 100)))


def best_spots(grid, lat, lon, max_km, n=5, min_sep_km=20):
    """Highest-scoring places within max_km, spread apart; ties broken by distance."""
    la = skyq.BBOX[2] - (np.arange(grid["score"].shape[0]) + 0.5) * RES
    lo = skyq.BBOX[1] + (np.arange(grid["score"].shape[1]) + 0.5) * RES
    LO, LA = np.meshgrid(lo[::4], la[::4])  # ponytail: ~9 km sampling is plenty for trip planning
    S, C = grid["score"][::4, ::4], grid["cls"][::4, ::4]
    D = skyq.haversine_km(lat, lon, LA, LO)
    ok = D <= max_km
    order = np.lexsort((D[ok], -np.round(S[ok])))
    cand = list(zip(S[ok][order], C[ok][order], LA[ok][order], LO[ok][order], D[ok][order]))
    picked = []
    for s, c, a, o, d in cand:
        if all(skyq.haversine_km(a, o, p["lat"], p["lon"]) >= min_sep_km for p in picked):
            picked.append(dict(lat=round(float(a), 3), lon=round(float(o), 3), score=round(float(s)), cls=int(c),
                               km=round(float(d))))
        if len(picked) == n:
            break
    return pd.DataFrame(picked)


def what_if(model, em_work, lat, lon, reduce_frac, radius_km=50):
    """Score at (lat, lon) after cutting unnecessary light by reduce_frac within radius_km (model re-run on the
    modified satellite image). Estimated scenario, not a physical measurement."""
    la = IRAQ_WORK[2] - (np.arange(em_work.shape[0]) + 0.5) * RES
    lo = IRAQ_WORK[1] + (np.arange(em_work.shape[1]) + 0.5) * RES
    LO, LA = np.meshgrid(lo, la)
    em2 = em_work.copy()
    em2[skyq.haversine_km(lat, lon, LA, LO) <= radius_km] *= 1 - reduce_frac
    # only the neighbourhood matters: predict on a small window around the site for speed
    win = (lat - 2.5, lon - 2.5, lat + 2.5, lon + 2.5)
    win = tuple(round(v / RES) * RES for v in win)
    sub = crop(em2, win, IRAQ_WORK)
    st = feature_stack(sub, win[2])
    r, c = int((win[2] - lat) / RES), int((lon - win[1]) / RES)
    p = model.predict_proba(pd.DataFrame({f: [st[f][r, c]] for f in FEATURES}))[0]
    return dict(score=float(p @ np.arange(4) / 3 * 100), cls=int(p.argmax()), proba=p.tolist())


if __name__ == "__main__":  # builds data, trains/compares, sanity checks
    import json
    m, rep = train_compare(rebuild=True)
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))
    grid = predict_grid(m, iraq_emission())
    b, d = at(grid, 33.312, 44.366), at(grid, 32.5, 41.5)
    print("Baghdad", b, "\nAnbar desert", d)
    assert b["score"] < d["score"] and b["cls"] <= 1 and d["cls"] >= 2
    print(best_spots(grid, 33.312, 44.366, 150))
    wi = what_if(m, iraq_emission(), 33.312, 44.366, 0.5)
    print("what-if Baghdad -50%:", wi)
    assert wi["score"] >= b["score"] - 1
    print("OK")
